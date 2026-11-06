"""API tests run in-process with FastAPI's TestClient (no server needed)."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app, state

from conftest import COMPLEX_TEXT, make_pdf


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Point the app at a throwaway vector store so tests never touch real data
    original_path = settings.vector_store_path
    settings.vector_store_path = str(tmp_path_factory.mktemp("vector_store"))
    try:
        with TestClient(app) as c:  # runs the startup lifespan
            yield c
    finally:
        settings.vector_store_path = original_path


@pytest.fixture
def empty_index(client):
    state.vector_store.clear()
    yield
    state.vector_store.clear()


@pytest.fixture(autouse=True)
def no_rate_limits(monkeypatch):
    from app.ratelimit import LLMRateLimiter
    monkeypatch.setattr(state, "rate_limiter", LLMRateLimiter(0, 0))


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_stats(client, empty_index):
    r = client.get("/stats")
    assert r.status_code == 200
    assert r.json()["total_chunks"] == 0


def test_readability(client):
    r = client.post("/readability", json={"text": COMPLEX_TEXT})
    assert r.status_code == 200
    data = r.json()
    assert data["is_patient_friendly"] is False
    assert "Consider simplifying" in data["recommendation"]


def test_upload_rejects_non_pdf(client):
    r = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_upload_pdf_then_ask(client, empty_index):
    pdf = make_pdf([
        "MEDICATIONS",
        "Take omeprazole 20mg by mouth once daily before breakfast.",
        "WARNING SIGNS",
        "Go to the emergency room if you have severe chest pain.",
    ])
    r = client.post("/upload", files={"file": ("discharge.pdf", pdf, "application/pdf")})
    assert r.status_code == 200
    assert r.json()["chunks_added"] > 0

    r = client.post("/ask", json={"question": "What medicine should I take?", "n_results": 1})
    assert r.status_code == 200
    data = r.json()
    assert "omeprazole" in data["answer"].lower()
    assert data["sources"][0]["source_file"] == "discharge.pdf"


def test_simplify_too_long_returns_413(client, monkeypatch):
    from app.generation.errors import InputTooLongError

    class TooLong:
        def simplify_text(self, text):
            raise InputTooLongError("Input is 9999 tokens")

    monkeypatch.setattr(state, "simplifier", TooLong())
    r = client.post("/simplify", json={"text": COMPLEX_TEXT})
    assert r.status_code == 413
    assert "9999 tokens" in r.json()["detail"]


def test_llm_endpoints_return_503_when_disabled(client, monkeypatch, empty_index):
    monkeypatch.setattr(settings, "enable_llm", False)
    assert client.get("/").json()["llm_enabled"] is False

    r = client.post("/simplify", json={"text": COMPLEX_TEXT})
    assert r.status_code == 503

    state.vector_store.add_chunks(
        [_chunk("Take omeprazole 20mg once daily.")], show_progress=False
    )
    r = client.post("/ask", json={"question": "What medicine?", "use_simplifier": True})
    assert r.status_code == 503
    # Plain retrieval still works without the LLM
    r = client.post("/ask", json={"question": "What medicine?", "use_simplifier": False})
    assert r.status_code == 200


def _chunk(text):
    from app.ingestion.chunker import TextChunk
    return TextChunk(content=text, chunk_id="t0", source_file="t.pdf",
                     chunk_index=0, token_count=8, section="medications")


class RecordingLLM:
    """Fake simplifier that records its inputs and echoes the placeholders."""

    def __init__(self):
        self.seen = []

    def simplify_text(self, text):
        from types import SimpleNamespace
        self.seen.append(text)
        return SimpleNamespace(simplified_text=f"Simple: {text}")

    def answer_question(self, question, context):
        self.seen += [question, context]
        return f"Answer for {question}\n{context}"


def test_simplify_never_sends_identifiers(client, monkeypatch):
    from test_deid import IDENTIFIERS, NOTE

    llm = RecordingLLM()
    monkeypatch.setattr(state, "simplifier", llm)
    r = client.post("/simplify", json={"text": NOTE})
    assert r.status_code == 200
    sent = "\n".join(llm.seen)
    leaked = [i for i in IDENTIFIERS if i in sent]
    assert not leaked, f"sent to the LLM: {leaked}"
    # The patient still gets their real details back
    data = r.json()
    assert data["simplified"] == f"Simple: {NOTE}"
    assert data["identifiers_removed"] >= 15


def test_ask_removes_header_names_from_other_chunks(client, monkeypatch, empty_index):
    llm = RecordingLLM()
    monkeypatch.setattr(state, "simplifier", llm)
    pdf = make_pdf([
        "Patient: Jane Roe    DOB: 02/14/1971",
        "MEDICATIONS",
        "Jane should take omeprazole 20mg once daily before breakfast.",
    ])
    assert client.post("/upload", files={"file": ("roe.pdf", pdf, "application/pdf")}).status_code == 200

    # Only the medications chunk is retrieved; it has no "Patient:" label
    r = client.post("/ask", json={"question": "What medicine do I take before breakfast?",
                                  "use_simplifier": True, "n_results": 1})
    assert r.status_code == 200
    sent = "\n".join(llm.seen)
    assert "omeprazole" in sent
    assert "Jane" not in sent and "Roe" not in sent and "02/14/1971" not in sent
    assert "Patient:" not in sent  # the header chunk wasn't retrieved
    assert "Jane should take omeprazole" in r.json()["answer"]
    assert r.json()["identifiers_removed"] == 1


def test_deidentify_can_be_disabled(client, monkeypatch):
    llm = RecordingLLM()
    monkeypatch.setattr(state, "simplifier", llm)
    monkeypatch.setattr(settings, "deidentify", False)
    r = client.post("/simplify", json={"text": "Patient: Jane Roe has GERD."})
    assert llm.seen == ["Patient: Jane Roe has GERD."]
    assert r.json()["identifiers_removed"] == 0


class Failing:
    def __init__(self, error):
        self.error = error

    def simplify_text(self, text):
        raise self.error


@pytest.mark.parametrize("error, status", [
    ("unavailable", 503),
    ("rejected", 502),
])
def test_provider_errors_map_to_http(client, monkeypatch, error, status):
    from app.generation.errors import LLMRequestError, LLMUnavailableError
    exc = (LLMUnavailableError if error == "unavailable" else LLMRequestError)("secret detail")
    monkeypatch.setattr(state, "simplifier", Failing(exc))
    r = client.post("/simplify", json={"text": COMPLEX_TEXT})
    assert r.status_code == status
    assert "secret detail" not in r.json()["detail"]  # provider details stay in the logs


def test_rate_limit_returns_429(client, monkeypatch):
    from app.ratelimit import LLMRateLimiter
    monkeypatch.setattr(state, "simplifier", RecordingLLM())
    monkeypatch.setattr(state, "rate_limiter", LLMRateLimiter(per_client_per_minute=2, per_day=0))
    codes = [client.post("/simplify", json={"text": "Take one pill."}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/simplify", json={"text": "Take one pill."})
    assert int(r.headers["Retry-After"]) >= 1
    # Endpoints that don't use the LLM aren't limited
    assert client.post("/readability", json={"text": COMPLEX_TEXT}).status_code == 200


def test_long_text_rejected_before_llm(client, monkeypatch):
    llm = RecordingLLM()
    monkeypatch.setattr(state, "simplifier", llm)
    r = client.post("/simplify", json={"text": "x" * (settings.max_input_chars + 1)})
    assert r.status_code == 413
    assert llm.seen == []


def test_n_results_is_bounded(client):
    r = client.post("/ask", json={"question": "What medicine?", "n_results": 1000})
    assert r.status_code == 422


def test_no_provider_returns_503(client, monkeypatch):
    monkeypatch.setattr(state, "simplifier", None)
    monkeypatch.setattr(state, "llm_error", "No LLM available")
    assert client.get("/").json()["llm_enabled"] is False
    r = client.post("/simplify", json={"text": COMPLEX_TEXT})
    assert r.status_code == 503


def test_simplify_reports_model(client, monkeypatch):
    from app.generation.providers import ProviderChain
    from test_gemini import Fake
    chain = ProviderChain([("gemini (test)", lambda: Fake("gemini-test"))])
    monkeypatch.setattr(state, "simplifier", chain)
    r = client.post("/simplify", json={"text": COMPLEX_TEXT})
    assert r.json()["model_used"] == "gemini-test"
    assert client.get("/").json()["llm_providers"] == ["gemini (test)"]


def test_ask_without_documents(client, empty_index):
    r = client.post("/ask", json={"question": "What medicine should I take?"})
    assert r.status_code == 400
