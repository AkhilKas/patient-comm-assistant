"""Pick the LLM provider(s) from settings: Gemini first, local model as fallback.

LLM_PROVIDER:
- "auto" (default): Gemini when GEMINI_API_KEY is set, with the local model
  as fallback if it is installed; the local model alone otherwise.
- "gemini": Gemini only (plus the local fallback if LLM_FALLBACK and installed).
- "local": the local transformers/MLX model only; nothing leaves the machine.

The lean install (requirements-core.txt, e.g. Render) has no local model, so
there a Gemini failure returns 503 instead of falling back.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
import threading
from typing import Callable

from .base import BaseSimplifier, SimplificationResult
from .errors import LLMUnavailableError


class LLMNotConfiguredError(RuntimeError):
    """No provider can run with the current settings and installed packages."""


def local_llm_available() -> bool:
    """True if a local model can be loaded (checked without importing torch)."""
    found = lambda name: importlib.util.find_spec(name) is not None  # noqa: E731
    mlx = sys.platform == "darwin" and platform.machine() == "arm64" and found("mlx_lm")
    return mlx or (found("torch") and found("transformers"))


class ProviderChain:
    """Same interface as a simplifier; tries each provider in order and moves
    on to the next only when one is unavailable (timeout, rate limit, outage).

    Providers are created on first use, so the local model is loaded only if
    it is actually needed.
    """

    def __init__(self, providers: list[tuple[str, Callable[[], BaseSimplifier]]]):
        if not providers:
            raise LLMNotConfiguredError("No LLM provider configured")
        self._factories = providers
        self._instances: dict[int, BaseSimplifier] = {}
        self._lock = threading.Lock()

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self._factories]

    def simplify_text(self, text: str) -> SimplificationResult:
        return self._call("simplify_text", text)

    def answer_question(self, question: str, context: str) -> str:
        return self._call("answer_question", question, context)

    def simplify_with_verification(self, text: str) -> SimplificationResult:
        return self._call("simplify_with_verification", text)

    def _provider(self, index: int) -> BaseSimplifier:
        with self._lock:
            if index not in self._instances:
                self._instances[index] = self._factories[index][1]()
            return self._instances[index]

    def _call(self, method: str, *args):
        failures = []
        for index, (label, _) in enumerate(self._factories):
            try:
                provider = self._provider(index)
            except Exception as e:  # e.g. the local model failed to load
                failures.append(f"{label} failed to load: {e}")
                continue
            try:
                return getattr(provider, method)(*args)
            except LLMUnavailableError as e:
                failures.append(f"{label}: {e}")
                if index + 1 < len(self._factories):
                    print(f"⚠ {label} unavailable ({e}); falling back to {self._factories[index + 1][0]}")
        raise LLMUnavailableError("; ".join(failures))


def build_provider_chain(settings) -> ProviderChain:
    provider = settings.llm_provider
    if provider not in ("auto", "gemini", "local"):
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; use 'auto', 'gemini' or 'local'")

    chain: list[tuple[str, Callable[[], BaseSimplifier]]] = []
    use_gemini = provider == "gemini" or (provider == "auto" and settings.gemini_api_key)
    if use_gemini:
        if not settings.gemini_api_key:
            raise LLMNotConfiguredError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")

        def gemini():
            from .gemini import GeminiSimplifier
            return GeminiSimplifier(
                api_key=settings.gemini_api_key,
                model_name=settings.gemini_model,
                timeout=settings.gemini_timeout,
            )
        chain.append((f"gemini ({settings.gemini_model})", gemini))

    want_local = provider == "local" or not use_gemini or settings.llm_fallback
    if want_local and local_llm_available():
        def local():
            from .simplifier import MedicalSimplifier
            return MedicalSimplifier(
                use_preset=settings.llm_preset,
                backend=settings.llm_backend,
                load_in_4bit=settings.load_in_4bit,
                hf_token=settings.hf_token,
            )
        chain.append((f"local ({settings.llm_preset})", local))

    if not chain:
        raise LLMNotConfiguredError(
            "No LLM available: set GEMINI_API_KEY, or install the local model "
            "dependencies (pip install -r requirements.txt)."
        )
    return ProviderChain(chain)
