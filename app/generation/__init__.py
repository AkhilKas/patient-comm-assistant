"""Generation module for medical text simplification.

The simplifier (and with it torch/transformers) is imported lazily, so the
API can run on a lean install without the LLM dependencies.
"""

from .errors import InputTooLongError
from .prompts import SYSTEM_PROMPT, SIMPLIFY_PROMPT, ANSWER_PROMPT

__all__ = [
    "MedicalSimplifier",
    "create_simplifier",
    "InputTooLongError",
    "SYSTEM_PROMPT",
    "SIMPLIFY_PROMPT",
    "ANSWER_PROMPT",
]


def __getattr__(name):
    if name in ("MedicalSimplifier", "create_simplifier"):
        from . import simplifier
        return getattr(simplifier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
