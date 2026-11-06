"""Provider-independent simplification logic (no torch imports).

Each provider (local transformers/MLX model, Gemini API) implements
`_generate`; the prompts and the two-stage flow are shared.
"""

from dataclasses import dataclass
from typing import Optional

from .prompts import SYSTEM_PROMPT, SIMPLIFY_PROMPT, ANSWER_PROMPT, READABILITY_CHECK_PROMPT


@dataclass
class SimplificationResult:
    """Container for simplification results."""
    original_text: str
    simplified_text: str
    model_used: str
    readability_before: Optional[dict] = None
    readability_after: Optional[dict] = None


class BaseSimplifier:
    """Prompting logic shared by all providers."""

    model_name: str = "unknown"

    def _generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        raise NotImplementedError

    def simplify_text(self, text: str) -> SimplificationResult:
        """
        Simplify medical text to 8th-grade reading level.

        Args:
            text: Medical text to simplify

        Returns:
            SimplificationResult with original and simplified text
        """
        prompt = SIMPLIFY_PROMPT.format(text=text)
        simplified = self._generate(prompt)

        return SimplificationResult(
            original_text=text,
            simplified_text=simplified,
            model_used=self.model_name
        )

    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a patient question using retrieved context.

        Args:
            question: Patient's question
            context: Retrieved context from vector store

        Returns:
            Simplified answer
        """
        prompt = ANSWER_PROMPT.format(question=question, context=context)
        return self._generate(prompt)

    def simplify_with_verification(self, text: str) -> SimplificationResult:
        """
        Two-stage simplification with readability verification.

        Stage 1: Initial simplification
        Stage 2: Check and re-simplify if needed
        """
        # Stage 1: Initial simplification
        result = self.simplify_text(text)

        # Stage 2: Verification (optional re-simplification)
        verification_prompt = READABILITY_CHECK_PROMPT.format(text=result.simplified_text)
        result.simplified_text = self._generate(verification_prompt)

        return result
