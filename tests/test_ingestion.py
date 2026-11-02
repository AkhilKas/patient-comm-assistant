"""Tests for PDF loading and chunking (no models needed)."""

from app.ingestion.chunker import MedicalTextChunker
from app.ingestion.pdf_loader import MedicalPDFLoader

from conftest import make_pdf


def test_chunker_splits_by_section(sample_chunks):
    sections = {c.section for c in sample_chunks}
    assert {"medications", "warning signs", "follow-up"} <= sections


def test_chunks_respect_size_limit(sample_document):
    chunker = MedicalTextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) > 1
    # Overlap can push a chunk slightly past the limit, but not far past it
    assert all(c.token_count <= 50 + 10 for c in chunks)


def test_chunk_ids_are_unique(sample_chunks):
    ids = [c.chunk_id for c in sample_chunks]
    assert len(ids) == len(set(ids))


def test_pdf_loader_extracts_and_expands_abbreviations(tmp_path):
    pdf_path = tmp_path / "discharge.pdf"
    pdf_path.write_bytes(make_pdf([
        "DISCHARGE INSTRUCTIONS",
        "MEDICATIONS",
        "Omeprazole 20mg po qd before breakfast.",
    ]))

    doc = MedicalPDFLoader().load(pdf_path)

    assert doc.metadata.page_count == 1
    assert doc.metadata.doc_type == "discharge_summary"
    assert "by mouth once daily" in doc.content
    assert "discharge instructions" in doc.metadata.sections_found
