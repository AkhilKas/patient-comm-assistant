"""Gemini provider and fallback chain, against a mocked HTTP transport."""

import json
from types import SimpleNamespace

import httpx
import pytest

from app.generation import providers
from app.generation.base import BaseSimplifier
from app.generation.errors import LLMRequestError, LLMUnavailableError
from app.generation.gemini import GeminiSimplifier
from app.generation.providers import LLMNotConfiguredError, ProviderChain, build_provider_chain

KEY = "test-key-not-real"


def gemini_with(handler, model="gemini-test"):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiSimplifier(api_key=KEY, model_name=model, client=client)


def ok(text, **extra):
    return httpx.Response(200, json={"candidates": [
        {"content": {"parts": [{"text": text}]}, "finishReason": "STOP", **extra}
    ]})


# ---------------------------------------------------------------- Gemini

def test_request_format():
    seen = {}

    def handler(request):
        seen["request"] = request
        return ok("Take one pill each morning.")

    result = gemini_with(handler).simplify_text("Administer 1 tablet PO qAM.")
    req = seen["request"]
    body = json.loads(req.content)

    assert req.method == "POST"
    assert req.url.path == "/v1beta/models/gemini-test:generateContent"
    assert req.headers["x-goog-api-key"] == KEY
    assert KEY not in str(req.url)
    assert "8th grader" in body["systemInstruction"]["parts"][0]["text"]
    assert "Administer 1 tablet PO qAM." in body["contents"][0]["parts"][0]["text"]
    assert body["generationConfig"]["maxOutputTokens"] > 0
    assert result.simplified_text == "Take one pill each morning."
    assert result.model_used == "gemini-test"


def test_thinking_parts_are_skipped():
    def handler(request):
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": "Let me think...", "thought": True},
            {"text": "Rest at home."},
        ]}}]})

    assert gemini_with(handler).answer_question("What now?", "Rest.") == "Rest at home."


@pytest.mark.parametrize("status", [408, 429, 500, 503, 504])
def test_retryable_status_is_unavailable(status):
    handler = lambda r: httpx.Response(status, json={"error": {"message": "busy"}})  # noqa: E731
    with pytest.raises(LLMUnavailableError, match=str(status)):
        gemini_with(handler).simplify_text("text")


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_rejected_request_is_not_retryable(status):
    handler = lambda r: httpx.Response(status, json={"error": {"message": "API key not valid"}})  # noqa: E731
    with pytest.raises(LLMRequestError, match="API key not valid") as exc:
        gemini_with(handler).simplify_text("text")
    assert KEY not in str(exc.value)


def test_timeout_and_network_errors_are_unavailable():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def offline(request):
        raise httpx.ConnectError("no route", request=request)

    for handler in (timeout, offline):
        with pytest.raises(LLMUnavailableError):
            gemini_with(handler).simplify_text("text")


@pytest.mark.parametrize("payload", [
    {"promptFeedback": {"blockReason": "SAFETY"}},
    {"candidates": []},
    {"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]},
])
def test_blocked_or_empty_answer_is_unavailable(payload):
    with pytest.raises(LLMUnavailableError):
        gemini_with(lambda r: httpx.Response(200, json=payload)).simplify_text("text")


def test_missing_key_is_rejected():
    with pytest.raises(ValueError):
        GeminiSimplifier(api_key="")


# ---------------------------------------------------------------- chain

class Fake(BaseSimplifier):
    def __init__(self, name, error=None):
        self.model_name, self.error, self.calls = name, error, 0

    def _generate(self, prompt, system_prompt=""):
        self.calls += 1
        if self.error:
            raise self.error
        return f"{self.model_name} answer"


def test_falls_back_when_primary_unavailable():
    primary, local = Fake("gemini", LLMUnavailableError("429")), Fake("local")
    chain = ProviderChain([("gemini", lambda: primary), ("local", lambda: local)])
    result = chain.simplify_text("text")
    assert result.simplified_text == "local answer"
    assert result.model_used == "local"
    assert chain.answer_question("q", "c") == "local answer"


def test_fallback_not_loaded_when_primary_works():
    loaded = []
    chain = ProviderChain([
        ("gemini", lambda: Fake("gemini")),
        ("local", lambda: loaded.append(1) or Fake("local")),
    ])
    assert chain.simplify_text("text").model_used == "gemini"
    assert loaded == []


def test_rejected_request_does_not_fall_back():
    local = Fake("local")
    chain = ProviderChain([
        ("gemini", lambda: Fake("gemini", LLMRequestError("bad key"))),
        ("local", lambda: local),
    ])
    with pytest.raises(LLMRequestError):
        chain.simplify_text("text")
    assert local.calls == 0


def test_fallback_that_fails_to_load_is_skipped():
    def broken():
        raise MemoryError("out of memory")

    chain = ProviderChain([("gemini", lambda: Fake("gemini", LLMUnavailableError("503"))),
                           ("local", broken)])
    with pytest.raises(LLMUnavailableError, match="local failed to load"):
        chain.simplify_text("text")


def test_providers_are_created_once():
    created = []
    chain = ProviderChain([("gemini", lambda: created.append(1) or Fake("gemini"))])
    chain.simplify_text("a")
    chain.simplify_text("b")
    assert created == [1]


# ---------------------------------------------------------------- settings

def cfg(**overrides):
    base = dict(llm_provider="auto", gemini_api_key=None, gemini_model="gemini-test",
                gemini_timeout=5.0, llm_fallback=True, llm_preset="phi3",
                llm_backend="auto", load_in_4bit=False, hf_token=None)
    return SimpleNamespace(**{**base, **overrides})


@pytest.mark.parametrize("overrides, local, expected", [
    ({"gemini_api_key": KEY}, True, ["gemini", "local"]),
    ({"gemini_api_key": KEY}, False, ["gemini"]),
    ({"gemini_api_key": KEY, "llm_fallback": False}, True, ["gemini"]),
    ({}, True, ["local"]),
    ({"llm_provider": "local", "gemini_api_key": KEY}, True, ["local"]),
    ({"llm_provider": "gemini", "gemini_api_key": KEY}, False, ["gemini"]),
])
def test_chain_from_settings(monkeypatch, overrides, local, expected):
    monkeypatch.setattr(providers, "local_llm_available", lambda: local)
    chain = build_provider_chain(cfg(**overrides))
    assert [label.split()[0] for label in chain.labels] == expected


@pytest.mark.parametrize("overrides, local", [
    ({}, False),                             # lean install, no key
    ({"llm_provider": "gemini"}, True),      # forced Gemini without a key
    ({"llm_provider": "local", "gemini_api_key": KEY}, False),
])
def test_not_configured(monkeypatch, overrides, local):
    monkeypatch.setattr(providers, "local_llm_available", lambda: local)
    with pytest.raises(LLMNotConfiguredError):
        build_provider_chain(cfg(**overrides))


def test_unknown_provider():
    with pytest.raises(ValueError):
        build_provider_chain(cfg(llm_provider="openai"))
