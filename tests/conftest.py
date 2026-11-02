"""Shared fixtures for the test suite."""

import os
import uuid

import pytest

from app.ingestion.chunker import MedicalTextChunker
from app.ingestion.pdf_loader import DocumentMetadata, LoadedDocument

SAMPLE_DISCHARGE = """
DISCHARGE SUMMARY

HOSPITAL COURSE:
You were admitted to the hospital after experiencing chest pain. We performed several tests including an EKG, blood tests for cardiac enzymes, and a stress test. All tests came back normal, and you did not have a heart attack.

Your symptoms were most likely caused by gastroesophageal reflux disease (GERD), also known as acid reflux. This happens when stomach acid flows back into your esophagus and can cause chest pain that feels similar to heart pain.

DIAGNOSIS:
1. Gastroesophageal Reflux Disease (GERD)
2. Chest pain, non-cardiac

MEDICATIONS:
1. Omeprazole 20mg - Take one tablet by mouth once daily, 30 minutes before breakfast. This medication reduces stomach acid and will help with your reflux symptoms.
2. Aspirin 81mg - Continue taking one tablet by mouth once daily for heart protection.
3. STOP taking ibuprofen (Advil, Motrin) or naproxen (Aleve) as these medications can worsen acid reflux.

DIET:
- Avoid spicy, fatty, or fried foods
- Do not eat within 3 hours of bedtime
- Limit caffeine and alcohol consumption

FOLLOW-UP:
Schedule an appointment with your primary care doctor within 1 week.

WARNING SIGNS:
Return to the emergency room right away if you experience:
- Severe chest pain or pressure that does not go away
- Chest pain that spreads to your arm, jaw, or back
- Difficulty breathing or shortness of breath
- Vomiting blood or material that looks like coffee grounds
"""

COMPLEX_TEXT = """
The patient presents with acute exacerbation of chronic obstructive
pulmonary disease, characterized by increased dyspnea, productive cough
with purulent sputum, and decreased exercise tolerance. Arterial blood
gas analysis reveals hypoxemia with compensated respiratory acidosis.
"""

SIMPLE_TEXT = """
Your lung problem got worse. You're having more trouble breathing
than usual and coughing up thick mucus. A blood test showed your
oxygen levels are low. We'll give you medicine to help you breathe better.
"""


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_LLM_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_LLM_TESTS=1 to run LLM tests")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def sample_document() -> LoadedDocument:
    return LoadedDocument(
        content=SAMPLE_DISCHARGE,
        metadata=DocumentMetadata(
            source_file="sample_discharge.pdf",
            page_count=1,
            doc_type="discharge_summary",
        ),
    )


@pytest.fixture
def sample_chunks(sample_document):
    return MedicalTextChunker(chunk_size=200, chunk_overlap=30).chunk_document(sample_document)


@pytest.fixture(scope="session")
def embedding_model():
    from app.retrieval.embeddings import get_embedding_model
    return get_embedding_model(use_preset="fast")


@pytest.fixture
def vector_store(embedding_model, sample_chunks):
    """In-memory store with the sample document indexed.

    Chroma's in-memory clients share state within a process, so each test
    gets a uniquely named collection.
    """
    from app.retrieval.vector_store import VectorStore
    store = VectorStore(
        collection_name=f"test_{uuid.uuid4().hex}",
        embedding_model=embedding_model,
    )
    store.add_chunks(sample_chunks, show_progress=False)
    return store


def make_pdf(lines: list[str]) -> bytes:
    """Build a minimal single-page text PDF without extra dependencies."""
    def escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    text_ops = "\n".join(f"({escape(line)}) Tj T*" for line in lines)
    stream = f"BT /F1 11 Tf 14 TL 50 750 Td\n{text_ops}\nET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, xref_pos
    )
    return bytes(out)
