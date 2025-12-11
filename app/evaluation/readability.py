"""Readability scoring for patient-friendly text verification."""

from dataclasses import dataclass
import textstat


@dataclass
class ReadabilityScore:
    """Container for readability metrics."""
    flesch_reading_ease: float      # 0-100, higher = easier (target: 60-70)
    flesch_kincaid_grade: float     # US grade level (target: <= 8)
    gunning_fog: float              # Grade level (target: <= 8)
    smog_index: float               # Grade level (target: <= 8)
    coleman_liau_index: float       # Grade level (target: <= 8)
    avg_grade_level: float          # Average of grade-level metrics
    is_patient_friendly: bool       # True if avg_grade <= 8
    word_count: int
    sentence_count: int
    avg_words_per_sentence: float
    
    def __str__(self) -> str:
        status = "✓ Patient-friendly" if self.is_patient_friendly else "✗ Too complex"
        return f"""
Readability Report ({status})
{'='*40}
Flesch Reading Ease:  {self.flesch_reading_ease:.1f}/100 (higher = easier)
Flesch-Kincaid Grade: {self.flesch_kincaid_grade:.1f} (target: ≤8)
Gunning Fog Index:    {self.gunning_fog:.1f} (target: ≤8)
SMOG Index:           {self.smog_index:.1f} (target: ≤8)
Coleman-Liau Index:   {self.coleman_liau_index:.1f} (target: ≤8)
{'='*40}
Average Grade Level:  {self.avg_grade_level:.1f}
{'='*40}
Word Count:           {self.word_count}
Sentence Count:       {self.sentence_count}
Avg Words/Sentence:   {self.avg_words_per_sentence:.1f}
"""

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "flesch_reading_ease": self.flesch_reading_ease,
            "flesch_kincaid_grade": self.flesch_kincaid_grade,
            "gunning_fog": self.gunning_fog,
            "smog_index": self.smog_index,
            "coleman_liau_index": self.coleman_liau_index,
            "avg_grade_level": self.avg_grade_level,
            "is_patient_friendly": self.is_patient_friendly,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "avg_words_per_sentence": self.avg_words_per_sentence,
        }


class ReadabilityScorer:
    """
    Readability scorer for medical text simplification.
    Uses multiple formulas and averages them for robust grade-level estimate.
    """
    
    def __init__(self, target_grade_level: float = 8.0):
        self.target_grade_level = target_grade_level
    
    def score(self, text: str) -> ReadabilityScore:
        """Calculate readability scores for text."""
        # Handle empty or very short text
        if not text or len(text.split()) < 10:
            return ReadabilityScore(
                flesch_reading_ease=100.0,
                flesch_kincaid_grade=0.0,
                gunning_fog=0.0,
                smog_index=0.0,
                coleman_liau_index=0.0,
                avg_grade_level=0.0,
                is_patient_friendly=True,
                word_count=len(text.split()) if text else 0,
                sentence_count=textstat.sentence_count(text) if text else 0,
                avg_words_per_sentence=0.0
            )
        
        # Calculate individual scores
        flesch_ease = textstat.flesch_reading_ease(text)
        flesch_grade = textstat.flesch_kincaid_grade(text)
        fog = textstat.gunning_fog(text)
        smog = textstat.smog_index(text)
        coleman = textstat.coleman_liau_index(text)
        
        # Calculate average grade level
        grade_scores = [flesch_grade, fog, smog, coleman]
        valid_scores = [s for s in grade_scores if 0 <= s <= 20]
        avg_grade = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        # Text stats
        word_count = textstat.lexicon_count(text, removepunct=True)
        sentence_count = textstat.sentence_count(text)
        avg_words_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        return ReadabilityScore(
            flesch_reading_ease=flesch_ease,
            flesch_kincaid_grade=flesch_grade,
            gunning_fog=fog,
            smog_index=smog,
            coleman_liau_index=coleman,
            avg_grade_level=avg_grade,
            is_patient_friendly=avg_grade <= self.target_grade_level,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_words_per_sentence=avg_words_sentence
        )
    
    def compare(self, original: str, simplified: str) -> dict:
        """Compare readability of original vs simplified text."""
        original_score = self.score(original)
        simplified_score = self.score(simplified)
        
        grade_improvement = original_score.avg_grade_level - simplified_score.avg_grade_level
        ease_improvement = simplified_score.flesch_reading_ease - original_score.flesch_reading_ease
        
        return {
            "original": original_score,
            "simplified": simplified_score,
            "grade_level_reduction": grade_improvement,
            "flesch_ease_improvement": ease_improvement,
            "met_target": simplified_score.is_patient_friendly,
            "summary": self._comparison_summary(original_score, simplified_score, grade_improvement)
        }
    
    def _comparison_summary(self, original: ReadabilityScore, simplified: ReadabilityScore, improvement: float) -> str:
        if simplified.is_patient_friendly:
            status = "✓ SUCCESS: Text is now patient-friendly"
        else:
            status = f"⚠ NEEDS WORK: Still at grade {simplified.avg_grade_level:.1f} (target: ≤{self.target_grade_level})"
        
        return f"""
Simplification Results
{'='*40}
{status}

Grade Level: {original.avg_grade_level:.1f} → {simplified.avg_grade_level:.1f} ({improvement:+.1f})
Flesch Ease: {original.flesch_reading_ease:.0f} → {simplified.flesch_reading_ease:.0f}
Words/Sentence: {original.avg_words_per_sentence:.1f} → {simplified.avg_words_per_sentence:.1f}
"""


def check_readability(text: str, target_grade: float = 8.0) -> ReadabilityScore:
    """Quick function to check readability of text."""
    scorer = ReadabilityScorer(target_grade_level=target_grade)
    return scorer.score(text)


def is_patient_friendly(text: str, target_grade: float = 8.0) -> bool:
    """Quick check if text is patient-friendly."""
    score = check_readability(text, target_grade)
    return score.is_patient_friendly


if __name__ == "__main__":
    scorer = ReadabilityScorer()
    
    complex_text = """
    The patient presents with acute exacerbation of chronic obstructive 
    pulmonary disease, characterized by increased dyspnea, productive cough 
    with purulent sputum, and decreased exercise tolerance. Arterial blood 
    gas analysis reveals hypoxemia with compensated respiratory acidosis.
    """
    
    simple_text = """
    Your lung problem got worse. You're having more trouble breathing 
    than usual and coughing up thick mucus. A blood test showed your 
    oxygen levels are low. We'll give you medicine to help you breathe better.
    """
    
    print("COMPLEX TEXT:")
    print(scorer.score(complex_text))
    
    print("\nSIMPLIFIED TEXT:")
    print(scorer.score(simple_text))
    
    print("\nCOMPARISON:")
    comparison = scorer.compare(complex_text, simple_text)
    print(comparison["summary"])