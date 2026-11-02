"""Compact CMU syllable table for textstat.

textstat counts syllables by looking words up in NLTK's CMU Pronouncing
Dictionary, which it loads as nested lists of phoneme strings for all
~134k words (about 57MB resident). It only ever uses the syllable count of
a word's first pronunciation, so we build {word: syllables} straight from
the dictionary file instead (about 17MB), with identical scores.

If textstat's internals change, this falls back to textstat's own loader.
"""

import logging
import os
from functools import cache

import nltk

logger = logging.getLogger(__name__)

CMUDICT = "corpora/cmudict/cmudict"


def ensure_cmudict() -> str:
    """Return the CMU dictionary path, downloading it if needed.

    Downloads to the first NLTK_DATA directory when that is set.
    """
    try:
        return str(nltk.data.find(CMUDICT))
    except LookupError:
        nltk_data = os.environ.get("NLTK_DATA", "").split(os.pathsep)[0] or None
        if nltk_data:
            os.makedirs(nltk_data, exist_ok=True)
        nltk.download("cmudict", download_dir=nltk_data, quiet=True)
        return str(nltk.data.find(CMUDICT))


class _SyllableTable(dict):
    """dict[word, syllables] that looks like textstat's cmudict to its reader.

    textstat reads cmu_dict[word][0] and counts phonemes ending in a stress
    digit, so each lookup returns one pronunciation of that many "phonemes".
    """

    def __getitem__(self, word):
        return [["1"] * dict.__getitem__(self, word)]


@cache
def syllable_table() -> _SyllableTable:
    counts: dict[str, int] = {}
    with open(ensure_cmudict(), encoding="latin-1") as f:
        for line in f:
            if line.startswith(";;;"):
                continue
            word, _variant, *phones = line.split()
            word = word.lower()
            if word not in counts:  # first pronunciation, as textstat uses [0]
                counts[word] = sum(1 for p in phones if p[-1].isdigit())
    return _SyllableTable(counts)


def install() -> bool:
    """Point textstat's syllable counter at the compact table."""
    try:
        from textstat.backend.counts import _count_syllables
        from textstat.backend.utils._get_lang_root import get_lang_root
    except ImportError:
        logger.warning("textstat internals changed; using its default CMU dictionary")
        return False

    original = _count_syllables.get_cmudict
    if getattr(original, "_compact", False):
        return True

    def get_cmudict(lang: str):
        return syllable_table() if get_lang_root(lang) == "en" else original(lang)

    get_cmudict._compact = True
    _count_syllables.get_cmudict = get_cmudict
    return True
