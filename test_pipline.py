"""
Test script for the ingestion -> retrieval pipeline.
Run from project root: python test_pipeline.py
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion.pdf_loader import LoadedDocument, DocumentMetadata
from app.ingestion.chunker import MedicalTextChunker
from app.retrieval.embeddings import get_embedding_model
from app.retrieval.vector_store import VectorStore


def create_sample_document() -> LoadedDocument:
    """Create a realistic sample discharge document."""
    content = """
DISCHARGE SUMMARY

Patient Name: [REDACTED]
Date of Discharge: December 2024
Admitting Diagnosis: Chest Pain, Rule Out Myocardial Infarction

HOSPITAL COURSE:
You were admitted to the hospital after experiencing chest pain. We performed several tests including an EKG, blood tests for cardiac enzymes, and a stress test. The good news is that all tests came back normal, and you did not have a heart attack.

Your symptoms were most likely caused by gastroesophageal reflux disease (GERD), also known as acid reflux. This happens when stomach acid flows back into your esophagus and can cause chest pain that feels similar to heart pain.

DISCHARGE DIAGNOSIS:
1. Gastroesophageal Reflux Disease (GERD)
2. Chest pain, non-cardiac

MEDICATIONS:
Please take the following medications as directed:

1. Omeprazole 20mg - Take one tablet by mouth once daily, 30 minutes before breakfast. This medication reduces stomach acid and will help with your reflux symptoms.

2. Aspirin 81mg - Continue taking one tablet by mouth once daily. This is for heart protection.

3. STOP taking ibuprofen (Advil, Motrin) or naproxen (Aleve) as these medications can worsen acid reflux and irritate your stomach.

DIET INSTRUCTIONS:
To help manage your acid reflux, please follow these dietary guidelines:
- Avoid spicy, fatty, or fried foods
- Do not eat within 3 hours of bedtime
- Limit caffeine and alcohol consumption
- Eat smaller, more frequent meals instead of large meals
- Avoid citrus fruits and tomato-based products if they trigger symptoms

ACTIVITY:
You may resume normal activities. Regular exercise is encouraged but avoid exercising immediately after eating.

FOLLOW-UP APPOINTMENTS:
1. Schedule an appointment with your primary care doctor within 1 week
2. If symptoms persist after 4 weeks of treatment, you may need an upper endoscopy (EGD) to examine your esophagus

WARNING SIGNS - RETURN TO EMERGENCY ROOM IF YOU EXPERIENCE:
- Severe chest pain or pressure that does not go away
- Chest pain that spreads to your arm, jaw, or back
- Difficulty breathing or shortness of breath
- Vomiting blood or material that looks like coffee grounds
- Black, tarry stools
- Fainting or severe dizziness
- Pain or difficulty swallowing

QUESTIONS:
If you have questions about your care, please call your doctor's office at [PHONE]. For emergencies, call 911 or go to your nearest emergency room.
"""
    
    return LoadedDocument(
        content=content,
        metadata=DocumentMetadata(
            source_file="sample_discharge.pdf",
            page_count=2,
            doc_type="discharge_summary",
            sections_found=["discharge instructions", "medications", "follow-up", "warning signs"]
        )
    )


def test_embeddings():
    """Test the embedding model."""
    print("\n" + "="*60)
    print("TESTING EMBEDDING MODEL")
    print("="*60)
    
    # Test with fast model (default)
    model = get_embedding_model(use_preset="fast")
    
    # Test single embedding
    text = "Take omeprazole 20mg once daily before breakfast"
    result = model.embed_text(text)
    print(f"\nSingle text embedding:")
    print(f"  Text: {text[:50]}...")
    print(f"  Dimensions: {result.dimensions}")
    print(f"  First 5 values: {result.embedding[:5]}")
    
    # Test similarity
    text1 = "What medications should I take?"
    text2 = "Take omeprazole 20mg daily"
    text3 = "The weather is nice today"
    
    sim1 = model.similarity(text1, text2)
    sim2 = model.similarity(text1, text3)
    
    print(f"\nSimilarity tests:")
    print(f"  '{text1}' vs '{text2}': {sim1:.3f}")
    print(f"  '{text1}' vs '{text3}': {sim2:.3f}")
    print(f"  (Higher = more similar)")
    
    return model


def test_chunking():
    """Test the chunking pipeline."""
    print("\n" + "="*60)
    print("TESTING CHUNKING")
    print("="*60)
    
    doc = create_sample_document()
    chunker = MedicalTextChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)
    
    print(f"\nDocument: {doc.metadata.source_file}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"\nChunks by section:")
    
    section_counts = {}
    for chunk in chunks:
        section = chunk.section or "unknown"
        section_counts[section] = section_counts.get(section, 0) + 1
    
    for section, count in section_counts.items():
        print(f"  {section}: {count} chunks")
    
    print(f"\nSample chunk:")
    print(f"  Section: {chunks[0].section}")
    print(f"  Tokens: {chunks[0].token_count}")
    print(f"  Content: {chunks[0].content[:150]}...")
    
    return chunks


def test_vector_store(chunks, embedding_model):
    """Test the vector store."""
    print("\n" + "="*60)
    print("TESTING VECTOR STORE")
    print("="*60)
    
    # Create in-memory store
    store = VectorStore(
        collection_name="test_discharge",
        embedding_model=embedding_model
    )
    
    # Add chunks
    store.add_chunks(chunks)
    
    print(f"\nStore stats:")
    print(f"  Total chunks: {store.get_chunk_count()}")
    print(f"  Sections: {store.get_all_sections()}")
    
    return store


def test_search(store):
    """Test search functionality."""
    print("\n" + "="*60)
    print("TESTING SEARCH")
    print("="*60)
    
    test_queries = [
        ("What medications do I need to take?", None),
        ("When should I go to the emergency room?", None),
        ("What foods should I avoid?", None),
        ("When is my follow-up appointment?", None),
        ("What caused my chest pain?", None),
    ]
    
    for query, section_filter in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        if section_filter:
            print(f"   Filter: section={section_filter}")
        print("-" * 50)
        
        results = store.search(query, n_results=2, filter_section=section_filter)
        
        for i, result in enumerate(results):
            print(f"\n   Result {i+1} (score: {result.score:.3f})")
            print(f"   Section: {result.metadata.get('section', 'unknown')}")
            # Truncate content for display
            content_preview = result.content[:120].replace('\n', ' ')
            print(f"   Content: {content_preview}...")


def test_section_search(store):
    """Test section-specific search."""
    print("\n" + "="*60)
    print("TESTING SECTION-SPECIFIC SEARCH")
    print("="*60)
    
    # Search only in medications section
    query = "How often should I take my medicine?"
    
    print(f"\n🔍 Query: \"{query}\"")
    print(f"   Searching in 'medications' section only")
    print("-" * 50)
    
    results = store.search_by_section(query, section="medications", n_results=2)
    
    if results:
        for i, result in enumerate(results):
            print(f"\n   Result {i+1} (score: {result.score:.3f})")
            content_preview = result.content[:150].replace('\n', ' ')
            print(f"   Content: {content_preview}...")
    else:
        print("   No results found in medications section")


def main():
    """Run all tests."""
    print("\n" + "🏥 PATIENT COMMUNICATION ASSISTANT - PIPELINE TEST " + "🏥")
    print("="*60)
    
    # Test embedding model
    embedding_model = test_embeddings()
    
    # Test chunking
    chunks = test_chunking()
    
    # Test vector store
    store = test_vector_store(chunks, embedding_model)
    
    # Test search
    test_search(store)
    
    # Test section-specific search
    test_section_search(store)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Add real PDF documents to data/sample_docs/")
    print("  2. Build the generation/simplification layer")
    print("  3. Add readability scoring")
    print("  4. Create FastAPI endpoints")


if __name__ == "__main__":
    main()