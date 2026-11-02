"""End-to-end tests with the local LLM. Skipped unless RUN_LLM_TESTS=1.

The first run downloads the model (several GB).
"""

import pytest

from app.evaluation.readability import ReadabilityScorer

from conftest import COMPLEX_TEXT

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def simplifier():
    from app.config import settings
    from app.generation.simplifier import MedicalSimplifier
    return MedicalSimplifier(
        use_preset=settings.llm_preset,
        backend=settings.llm_backend,
        hf_token=settings.hf_token,
    )


def test_simplify_lowers_grade_level(simplifier):
    result = simplifier.simplify_text(COMPLEX_TEXT)
    assert result.simplified_text
    comparison = ReadabilityScorer().compare(COMPLEX_TEXT, result.simplified_text)
    assert comparison["grade_level_reduction"] > 0


def test_answer_question_uses_context(simplifier, vector_store):
    question = "What medicine do I take for my stomach?"
    context = "\n\n".join(r.content for r in vector_store.search(question, n_results=2))
    answer = simplifier.answer_question(question, context)
    assert "omeprazole" in answer.lower()
