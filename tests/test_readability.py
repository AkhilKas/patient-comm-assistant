"""Tests for readability scoring (no models needed)."""

from app.evaluation.readability import ReadabilityScorer, is_patient_friendly

from conftest import COMPLEX_TEXT, SIMPLE_TEXT


def test_complex_text_is_not_patient_friendly():
    score = ReadabilityScorer().score(COMPLEX_TEXT)
    assert score.avg_grade_level > 8
    assert not score.is_patient_friendly


def test_simple_text_is_patient_friendly():
    assert is_patient_friendly(SIMPLE_TEXT)


def test_compare_reports_improvement():
    result = ReadabilityScorer().compare(COMPLEX_TEXT, SIMPLE_TEXT)
    assert result["grade_level_reduction"] > 0
    assert result["flesch_ease_improvement"] > 0
    assert result["met_target"]


def test_compact_syllable_table_matches_cmudict():
    import nltk
    from textstat.backend.counts import _count_syllables
    from app.evaluation.syllables import ensure_cmudict, syllable_table

    assert getattr(_count_syllables.get_cmudict, "_compact", False), "patch not installed"

    ensure_cmudict()
    full = nltk.corpus.cmudict.dict()
    table = syllable_table()
    words = {w.strip(".,:;()-").lower() for w in (COMPLEX_TEXT + SIMPLE_TEXT).split()}
    words = [w for w in words if w in full]
    assert len(words) > 40
    for w in words:
        expected = sum(1 for p in full[w][0] if p[-1].isdigit())
        assert sum(1 for p in table[w][0] if p[-1].isdigit()) == expected, w


def test_short_text_short_circuits():
    score = ReadabilityScorer().score("Take one pill.")
    assert score.avg_grade_level == 0.0
    assert score.word_count == 3
