"""Generation module for medical text simplification."""

from .simplifier import MedicalSimplifier, create_simplifier
from .prompts import SYSTEM_PROMPT, SIMPLIFY_PROMPT, ANSWER_PROMPT

__all__ = [
    "MedicalSimplifier",
    "create_simplifier",
    "SYSTEM_PROMPT",
    "SIMPLIFY_PROMPT",
    "ANSWER_PROMPT",
]
