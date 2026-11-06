"""Rule-based de-identification of text before it is sent to an LLM.

Identifiers are replaced with numbered placeholders ([NAME_1], [DATE_2], ...)
and the mapping is kept in memory so the real values can be put back into the
model's answer. Nothing here needs a model, so it adds no memory to the lean
server.

What is detected:
- Labelled fields: "Patient:", "Name:", "DOB:", "MRN:", "Address:", "Phone:",
  "Attending:", ... (the value is taken up to the next label or end of line).
- Fixed-format identifiers: emails, URLs, IP addresses, SSNs, phone/fax
  numbers, dates, street addresses, "City, ST 12345", long ID numbers and
  ages over 89 (from HIPAA's Safe Harbor list).
- Titled names: "Dr. Patel", "Mrs. Jane Doe", "Jane Doe, MD".
- Names found in labelled fields are also replaced wherever else they appear
  ("Patient: John Smith" also removes a later "Mr. Smith" or "John").

Limits: names with no label, title or earlier mention ("your daughter Maria")
are not detected. This lowers the risk of sending identifiers to an LLM; it
is not a guarantee and does not make the app HIPAA compliant.

Run on a file to see what would be sent:  python -m app.privacy note.pdf
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- labelled fields

_NAME_LABELS = [
    r"patient(?:'s)?\s+name", r"patient", r"full\s+name", r"name",
    r"attending(?:\s+physician)?", r"physician", r"provider", r"surgeon",
    r"referring(?:\s+physician)?", r"primary\s+care(?:\s+(?:physician|provider))?", r"pcp",
    r"emergency\s+contact", r"next\s+of\s+kin", r"guardian", r"caregiver",
]
_DATE_LABELS = [
    r"date\s+of\s+birth", r"dob", r"birth\s*date",
    r"(?:admission|admit|discharge|visit|service)\s+date",
    r"date\s+of\s+(?:admission|discharge|service|visit)",
]
_ID_LABELS = [
    r"mrn", r"medical\s+record(?:\s+(?:number|no\.?|#))?", r"account(?:\s+(?:number|no\.?))?",
    r"acct\.?", r"ssn", r"social\s+security(?:\s+number)?",
    r"(?:insurance|member|policy|subscriber|patient)\s+id", r"encounter(?:\s+(?:number|id))?",
    r"health\s+plan(?:\s+(?:number|id))?",
]
_ADDRESS_LABELS = [r"(?:home\s+|street\s+|mailing\s+)?address"]
_PHONE_LABELS = [r"phone", r"tel(?:ephone)?", r"cell", r"mobile", r"fax"]
_EMAIL_LABELS = [r"e-?mail"]

_LABEL_KINDS = (
    [(p, "NAME") for p in _NAME_LABELS]
    + [(p, "DATE") for p in _DATE_LABELS]
    + [(p, "ID") for p in _ID_LABELS]
    + [(p, "ADDRESS") for p in _ADDRESS_LABELS]
    + [(p, "PHONE") for p in _PHONE_LABELS]
    + [(p, "EMAIL") for p in _EMAIL_LABELS]
)
# Longest alternatives first so "patient name" wins over "patient"
_LABEL_KINDS.sort(key=lambda lk: -len(lk[0]))
_LABEL_GROUPS = {f"L{i}": kind for i, (_, kind) in enumerate(_LABEL_KINDS)}
# A bare "Name" only counts at the start of a line or after punctuation, so
# "Medication Name: Omeprazole" is not mistaken for a person.
_LABEL_RE = re.compile(
    r"(?im)(?<![\w-])(?:"
    + "|".join(
        (rf"(?<![A-Za-z] )(?P<L{i}>{p})" if p == "name" else rf"(?P<L{i}>{p})")
        for i, (p, _) in enumerate(_LABEL_KINDS)
    )
    + r")\s*(?:#|:|no\.)\s*:?[ \t]*"
)

# ---------------------------------------------------------------- patterns

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
          r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_STREET = (r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|"
           r"Way|Place|Pl|Terrace|Ter|Circle|Cir|Parkway|Pkwy|Highway|Hwy)")
_TITLE = r"(?:Dr|Mr|Mrs|Ms|Miss|Mx|Prof)"
_DEGREE = r"(?:M\.?D|D\.?O|R\.?N|N\.?P|P\.?A(?:-C)?|Ph\.?D|PharmD|DNP|APRN|MBBS)"
# "O'Brien" is one word; the "'s" in "Patel's" is not part of the name
_NAME_WORD = r"[A-Z](?:'[A-Z])?[a-zA-Z\-]+"

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("URL", re.compile(r"\b(?:https?://|www\.)\S+[^\s.,;:)]", re.I)),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),  # SSN
    ("PHONE", re.compile(r"(?<![\w-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")),
    # 7-digit numbers only next to a phone word, so dose ranges like 500-1000 survive
    ("PHONE", re.compile(r"(?i)(?<=call |text )\d{3}-\d{4}\b|(?<=\bat )\d{3}-\d{4}\b")),
    ("DATE", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})\b")),
    ("DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("DATE", re.compile(rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b")),
    ("DATE", re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\.?,?\s+\d{{4}}\b")),
    ("DATE", re.compile(rf"\b{_MONTH}\.?\s+\d{{4}}\b")),
    ("ADDRESS", re.compile(rf"\b\d{{1,6}}[ \t]+(?:{_NAME_WORD}[ \t]+){{1,4}}{_STREET}\b\.?"
                           r"(?:,?[ \t]+(?:Apt|Unit|Suite|Ste|#)\.?[ \t]*\w+)?")),
    ("ADDRESS", re.compile(rf"\b(?:{_NAME_WORD}[ \t]){{0,2}}{_NAME_WORD},[ \t]*[A-Z]{{2}}[ \t]+\d{{5}}(?:-\d{{4}})?\b")),
    ("ID", re.compile(r"\b\d{7,}\b")),
    ("AGE", re.compile(r"(?i)\b(?:9\d|1[0-1]\d)[- ](?:years?|yrs?|y/?o)(?:[- ]old)?\b")),
    ("AGE", re.compile(r"(?i)(?<=\bage )(?:9\d|1[0-1]\d)\b")),
    ("NAME", re.compile(rf"\b{_TITLE}\.?[ \t]+{_NAME_WORD}(?:[ \t]+{_NAME_WORD})?")),
    ("NAME", re.compile(rf"\b{_NAME_WORD}(?:[ \t]+[A-Z]\.)?[ \t]+{_NAME_WORD},?[ \t]+{_DEGREE}\b\.?")),
]

_PLACEHOLDER_RE = re.compile(
    r"\[?\b(NAME|DATE|PHONE|EMAIL|URL|IP|ID|ADDRESS|AGE)_(\d+)\b\]?", re.I
)
_TITLES = {"dr", "mr", "mrs", "ms", "miss", "mx", "prof"}
_NOT_NAMES = {"md", "do", "rn", "np", "pa", "phd", "jr", "sr", "ii", "iii", "the",
              "patient", "unknown", "none", "self", "and"}


# ---------------------------------------------------------------- core

def find_names(text: str) -> list[str]:
    """Names given in labelled fields ("Patient: John Smith"), plus their parts.

    Stored with each uploaded document so names can be removed from chunks
    that don't contain the header.
    """
    names: list[str] = []
    for kind, _, _, value in _labelled_values(text):
        if kind == "NAME":
            names.extend(_name_variants(value))
    return list(dict.fromkeys(names))


@dataclass
class Deidentifier:
    """Replaces identifiers with placeholders and puts them back afterwards.

    Use one instance per request so the same value gets the same placeholder
    in the question, the context and the answer.
    """
    known_names: list[str] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original
    _by_value: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)
    _counts: dict[str, int] = field(default_factory=dict, repr=False)

    def redact(self, text: str) -> str:
        names = list(self.known_names) + find_names(text)
        spans = self._spans(text, names)
        out, pos = [], 0
        for start, end, kind in spans:
            out.append(text[pos:start])
            out.append(self._placeholder(kind, text[start:end]))
            pos = end
        out.append(text[pos:])
        return "".join(out)

    def restore(self, text: str) -> str:
        def sub(m: re.Match) -> str:
            key = f"[{m.group(1).upper()}_{m.group(2)}]"
            return self.mapping.get(key, m.group(0))
        return _PLACEHOLDER_RE.sub(sub, text)

    @property
    def count(self) -> int:
        """Number of distinct identifiers replaced so far."""
        return len(self.mapping)

    def _placeholder(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key not in self._by_value:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            placeholder = f"[{kind}_{self._counts[kind]}]"
            self._by_value[key] = placeholder
            self.mapping[placeholder] = value
        return self._by_value[key]

    def _spans(self, text: str, names: list[str]) -> list[tuple[int, int, str]]:
        found: list[tuple[int, int, str]] = []
        for kind, start, end, _ in _labelled_values(text):
            found.append((start, end, kind))
        for kind, pattern in _PATTERNS:
            found.extend((m.start(), m.end(), kind) for m in pattern.finditer(text))
        name_re = _names_pattern(names)
        if name_re:
            found.extend((m.start(), m.end(), "NAME") for m in name_re.finditer(text))

        # Keep the earliest, then longest, span where detections overlap
        found.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        kept: list[tuple[int, int, str]] = []
        for span in found:
            if span[1] > span[0] and (not kept or span[0] >= kept[-1][1]):
                kept.append(span)
        return kept


def _labelled_values(text: str):
    """Yield (kind, start, end, value) for each "Label: value" in the text."""
    labels = list(_LABEL_RE.finditer(text))
    for i, m in enumerate(labels):
        kind = _LABEL_GROUPS[m.lastgroup]
        start = m.end()
        line_end = text.find("\n", start)
        line_end = len(text) if line_end == -1 else line_end
        nxt = labels[i + 1].start() if i + 1 < len(labels) else len(text)
        end = min(line_end, nxt)

        if not text[start:end].strip() and end == line_end and nxt > line_end:
            # "PATIENT NAME:" with the value on the next line
            nl_start = line_end + 1
            nl_end = text.find("\n", nl_start)
            nl_end = len(text) if nl_end == -1 else nl_end
            candidate = text[nl_start:min(nl_end, nxt)]
            if 0 < len(candidate.split()) <= 6:
                start, end = nl_start, min(nl_end, nxt)

        if kind == "NAME":
            end = start + _name_length(text[start:end])
        value = text[start:end]
        stripped = value.rstrip(" \t,;|")
        start += len(value) - len(value.lstrip())
        end = start + len(stripped.strip())
        if end > start:
            yield kind, start, end, text[start:end]


def _name_length(value: str) -> int:
    """Stop a labelled name at the end of its sentence ("Jane Roe. Call...")
    without cutting at initials or titles ("John A. Smith", "Dr. Patel")."""
    for m in re.finditer(r"\.(?=\s|$)", value):
        word = re.search(r"([A-Za-z]+)$", value[:m.start()])
        word = word.group(1) if word else ""
        if len(word) > 1 and word.lower() not in _TITLES:
            return m.start()
    return len(value)


def _name_variants(value: str) -> list[str]:
    """'Smith, John A., MD' -> ['Smith, John A., MD', 'Smith', 'John']."""
    variants = [value.strip()]
    for word in re.findall(r"[A-Za-z](?:'[A-Za-z])?[A-Za-z\-]+", value):
        if len(word) >= 2 and word[0].isupper() and word.lower() not in _TITLES | _NOT_NAMES:
            variants.append(word)
    return variants


def _names_pattern(names: list[str]) -> re.Pattern | None:
    """Match each name as written, in Title case or in UPPER case (not lower,
    so a patient called Will doesn't remove "you will")."""
    forms = set()
    for name in names:
        name = name.strip()
        if len(name) < 2:
            continue
        forms.update({name, name.title(), name.upper()})
    forms = {f for f in forms if not f.islower()}
    if not forms:
        return None
    alternation = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.compile(rf"(?<![\w-])(?:{alternation})(?![\w-])")


def main(argv: list[str]) -> int:
    import sys
    from pathlib import Path

    if len(argv) != 1:
        print("usage: python -m app.privacy <file.pdf|file.txt|->", file=sys.stderr)
        return 2
    src = argv[0]
    if src == "-":
        text = sys.stdin.read()
    elif src.lower().endswith(".pdf"):
        from app.ingestion.pdf_loader import MedicalPDFLoader
        text = MedicalPDFLoader().load(src).content
    else:
        text = Path(src).read_text()

    deid = Deidentifier()
    print(deid.redact(text))
    print(f"\n--- {deid.count} identifiers replaced ---", file=sys.stderr)
    for placeholder, original in deid.mapping.items():
        print(f"{placeholder:<14} {original}", file=sys.stderr)
    return 0

