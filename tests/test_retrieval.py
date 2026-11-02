"""Tests for embeddings and vector search (downloads the ~90MB embedding model once)."""

import pytest


def test_embedding_dimensions(embedding_model):
    result = embedding_model.embed_text("Take omeprazole 20mg once daily")
    assert result.dimensions == 384
    assert len(result.embedding) == 384


def test_lean_onnx_matches_chroma_default(embedding_model):
    import numpy as np
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    texts = ["Take omeprazole 20mg once daily.", "Call 911 for chest pain."] * 3
    default = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
    default.DOWNLOAD_PATH = embedding_model._onnx.DOWNLOAD_PATH
    expected = np.asarray(default(texts))
    actual = np.array([r.embedding for r in embedding_model.embed_batch(texts, show_progress=False)])
    assert np.abs(actual - expected).max() < 1e-5


def test_related_texts_are_more_similar(embedding_model):
    query = "What medications should I take?"
    related = embedding_model.similarity(query, "Take omeprazole 20mg daily")
    unrelated = embedding_model.similarity(query, "The weather is nice today")
    assert related > unrelated


def test_store_indexes_all_chunks(vector_store, sample_chunks):
    assert vector_store.get_chunk_count() == len(sample_chunks)
    assert "medications" in vector_store.get_all_sections()


@pytest.mark.parametrize("query, expected_section", [
    ("What medications do I need to take?", "medications"),
    ("When should I go to the emergency room?", "warning signs"),
    ("What foods should I avoid?", "diet"),
])
def test_search_finds_relevant_section(vector_store, query, expected_section):
    results = vector_store.search(query, n_results=2)
    assert expected_section in {r.metadata["section"] for r in results}


def test_search_by_section_filters(vector_store):
    results = vector_store.search_by_section("How often?", section="medications", n_results=2)
    assert results
    assert all(r.metadata["section"] == "medications" for r in results)


def test_clear_empties_store(vector_store):
    vector_store.clear()
    assert vector_store.get_chunk_count() == 0
