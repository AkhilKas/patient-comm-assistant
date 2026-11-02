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
    from app.generation.simplifier import InputTooLongError

    class TooLong:
        def simplify_text(self, text):
            raise InputTooLongError("Input is 9999 tokens")

    monkeypatch.setattr(state, "simplifier", TooLong())
    r = client.post("/simplify", json={"text": COMPLEX_TEXT})
    assert r.status_code == 413
    assert "9999 tokens" in r.json()["detail"]


def test_ask_without_documents(client, empty_index):
    r = client.post("/ask", json={"question": "What medicine should I take?"})
    assert r.status_code == 400
