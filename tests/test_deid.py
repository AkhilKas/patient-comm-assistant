"""De-identification: no identifier from a synthetic note may reach the LLM."""

import pytest

from app.privacy import Deidentifier, find_names

# Synthetic data only: every identifier below is made up.
NOTE = """DISCHARGE SUMMARY
Patient: John A. Smith    DOB: 04/12/1958    MRN: 00482913
Address: 42 Elm Street, Apt 3, Springfield, MA 01103
Phone: (413) 555-0147   Email: jsmith58@example.com
Attending: Dr. Priya Patel, MD
Emergency Contact: Mary Smith (wife) 413-555-0199
Admission Date: March 3, 2026
SSN: 123-45-6789

HOSPITAL COURSE:
Mr. Smith is a 67-year-old man admitted with chest pain. John tolerated the
stress test well. His mother, age 93, lives with him.

MEDICATIONS:
Medication Name: Omeprazole 20mg once daily before breakfast.
Take 500-1000 mg acetaminophen every 6 hours as needed. You will take aspirin 81mg daily.

FOLLOW-UP:
See Dr. Patel on 03/17/2026. Call 555-0123 with questions.
Portal: https://portal.example.org/jsmith
"""

IDENTIFIERS = [
    "John", "Smith", "04/12/1958", "00482913", "42 Elm Street", "Springfield",
    "01103", "555-0147", "jsmith58@example.com", "Priya", "Patel", "Mary",
    "555-0199", "March 3, 2026", "123-45-6789", "93", "03/17/2026", "555-0123",
    "portal.example.org",
]

# Clinical content that must survive
KEPT = [
    "Omeprazole 20mg", "500-1000 mg", "every 6 hours", "aspirin 81mg",
    "You will take", "67-year-old", "chest pain", "Medication Name:",
]


@pytest.fixture
def redacted():
    deid = Deidentifier()
    return deid, deid.redact(NOTE)


@pytest.mark.parametrize("identifier", IDENTIFIERS)
def test_identifier_removed(redacted, identifier):
    _, text = redacted
    assert identifier not in text


@pytest.mark.parametrize("phrase", KEPT)
def test_clinical_content_kept(redacted, phrase):
    _, text = redacted
    assert phrase in text


def test_restore_round_trips(redacted):
    deid, text = redacted
    assert deid.restore(text) == NOTE


def test_same_value_same_placeholder():
    deid = Deidentifier()
    a = deid.redact("Call Dr. Patel at (413) 555-0147.")
    b = deid.redact("Dr. Patel's number is (413) 555-0147.")
    assert a == "Call [NAME_1] at [PHONE_1]."
    assert b == "[NAME_1]'s number is [PHONE_1]."
    assert deid.count == 2


def test_restore_tolerates_llm_rewrites():
    deid = Deidentifier()
    deid.redact("Follow up with Dr. Patel on 03/17/2026.")
    answer = "See [name_1] on DATE_1. Unknown [NAME_9] stays."
    assert deid.restore(answer) == "See Dr. Patel on 03/17/2026. Unknown [NAME_9] stays."


def test_known_names_cover_chunks_without_header():
    names = find_names(NOTE)
    assert {"Smith", "John", "Patel", "Mary"} <= set(names)
    body = "John should rest. MR. SMITH may shower tomorrow. You will feel better."
    text = Deidentifier(known_names=names).redact(body)
    assert "John" not in text and "SMITH" not in text
    assert "You will feel better" in text


def test_label_value_on_next_line():
    text = Deidentifier().redact("PATIENT NAME:\nRoe, Jane\nDIAGNOSIS:\nGERD")
    assert "Roe" not in text and "Jane" not in text
    assert "GERD" in text


def test_medication_name_is_not_a_person():
    names = find_names("Medication Name: Omeprazole\nDrug name: Aspirin")
    assert names == []


def test_plain_clinical_text_unchanged():
    from conftest import SAMPLE_DISCHARGE
    deid = Deidentifier()
    assert deid.redact(SAMPLE_DISCHARGE) == SAMPLE_DISCHARGE
    assert deid.count == 0
