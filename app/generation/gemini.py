"""Gemini API provider (REST via httpx, no Google SDK).

Text reaches this module only after de-identification (see app/privacy).
On Gemini's free tier, Google may use prompts and responses to improve its
products, so use sample or de-identified documents only.
"""

from __future__ import annotations

from typing import Optional

import httpx

from .base import BaseSimplifier
from .errors import LLMRequestError, LLMUnavailableError
from .prompts import SYSTEM_PROMPT

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Worth trying a fallback provider: timeouts, rate limits and outages
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class GeminiSimplifier(BaseSimplifier):
    """Simplifier backed by the Gemini generateContent endpoint."""

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        max_output_tokens: int = 2048,
        timeout: float = 30.0,
        client: Optional[httpx.Client] = None,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        # The key goes in a header, never in the URL, so it can't end up in
        # error messages or access logs
        self._client = client or httpx.Client(timeout=timeout)
        self._headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    def _generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            # temperature/topP are ignored by current Gemini models
            "generationConfig": {"maxOutputTokens": self.max_output_tokens},
        }
        url = f"{API_BASE}/models/{self.model_name}:generateContent"
        try:
            response = self._client.post(url, json=body, headers=self._headers)
        except httpx.TimeoutException as e:
            raise LLMUnavailableError("Gemini timed out") from e
        except httpx.TransportError as e:
            raise LLMUnavailableError(f"Could not reach Gemini ({type(e).__name__})") from e

        if response.status_code != 200:
            message = _error_message(response)
            if response.status_code in RETRYABLE_STATUS:
                raise LLMUnavailableError(f"Gemini returned {response.status_code}: {message}")
            raise LLMRequestError(f"Gemini rejected the request ({response.status_code}): {message}")

        return _extract_text(response.json())


def _extract_text(data: dict) -> str:
    block = (data.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise LLMUnavailableError(f"Gemini blocked the prompt ({block})")

    candidates = data.get("candidates") or []
    if not candidates:
        raise LLMUnavailableError("Gemini returned no answer")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    # Skip the model's thinking, keep only the answer
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        reason = candidate.get("finishReason", "unknown")
        raise LLMUnavailableError(f"Gemini returned an empty answer (finishReason {reason})")
    return text


def _error_message(response: httpx.Response) -> str:
    try:
        return response.json()["error"]["message"]
    except Exception:
        return response.text[:200] or response.reason_phrase
