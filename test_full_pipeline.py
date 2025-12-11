"""
Full end-to-end test: Ingest → Retrieve → Simplify → Evaluate
Run from project root: python test_full_pipeline.py

NOTE: First run will download the LLM (~2-8GB depending on model)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion.pdf_loader import LoadedDocument, DocumentMetadata
from app.ingestion.chunker import MedicalTextChunker
from app.retrieval.vector_store import VectorStore
from app.retrieval.embeddings import get_embedding_model
from app.evaluation.readability import ReadabilityScorer, check_readability


def create_sample_document() -> LoadedDocument:
    """Create a sample discharge document."""
    content = """
DISCHARGE SUMMARY

HOSPITAL COURSE:
You were admitted to the hospital after experiencing chest pain. We performed several tests including an EKG, blood tests for cardiac enzymes, and a stress test. All tests came back normal, and you did not have a heart attack.

Your symptoms were most likely caused by gastroesophageal reflux disease (GERD), also known as acid reflux. This happens when stomach acid flows back into your esophagus and can cause chest pain that feels similar to heart pain.

DISCHARGE DIAGNOSIS:
1. Gastroesophageal Reflux Disease (GERD)
2. Chest pain, non-cardiac

MEDICATIONS:
Please take the following medications as directed:

1. Omeprazole 20mg - Take one tablet by mouth once daily, 30 minutes before breakfast. This medication reduces stomach acid and will help with your reflux symptoms.

2. Aspirin 81mg - Continue taking one tablet by mouth once daily for heart protection.

3. STOP taking ibuprofen (Advil, Motrin) or naproxen (Aleve) as these medications can worsen acid reflux.

WARNING SIGNS - RETURN TO EMERGENCY ROOM IF YOU EXPERIENCE:
- Severe chest pain or pressure that does not go away
- Chest pain that spreads to your arm, jaw, or back
- Difficulty breathing or shortness of breath
- Vomiting blood or material that looks like coffee grounds
- Black, tarry stools
"""
    return LoadedDocument(
        content=content,
        metadata=DocumentMetadata(
            source_file="sample_discharge.pdf",
            page_count=1,
            doc_type="discharge_summary"
        )
    )


def test_readability_only():
    """Test just the readability scorer (no LLM needed)."""
    print("\n" + "="*60)
    print("TESTING READABILITY SCORER")
    print("="*60)
    
    scorer = ReadabilityScorer(target_grade_level=8.0)
    
    # Complex medical text
    complex_text = """
    The patient presents with acute exacerbation of chronic obstructive 
    pulmonary disease, characterized by increased dyspnea, productive cough 
    with purulent sputum, and decreased exercise tolerance. Arterial blood 
    gas analysis reveals hypoxemia with compensated respiratory acidosis.
    """
    
    # Simple text
    simple_text = """
    Your lung problem got worse. You're having more trouble breathing 
    than usual and coughing up thick mucus. A blood test showed your 
    oxygen levels are low. We'll give you medicine to help you breathe better.
    """
    
    print("\n📊 Complex Medical Text:")
    complex_score = scorer.score(complex_text)
    print(f"   Grade Level: {complex_score.avg_grade_level:.1f}")
    print(f"   Patient-friendly: {complex_score.is_patient_friendly}")
    
    print("\n📊 Simplified Text:")
    simple_score = scorer.score(simple_text)
    print(f"   Grade Level: {simple_score.avg_grade_level:.1f}")
    print(f"   Patient-friendly: {simple_score.is_patient_friendly}")
    
    print("\n📊 Comparison:")
    comparison = scorer.compare(complex_text, simple_text)
    print(comparison["summary"])
    
    return scorer


def test_retrieval_pipeline():
    """Test ingestion and retrieval (no LLM needed)."""
    print("\n" + "="*60)
    print("TESTING RETRIEVAL PIPELINE")
    print("="*60)
    
    # Create document and chunks
    doc = create_sample_document()
    chunker = MedicalTextChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)
    print(f"\n✓ Created {len(chunks)} chunks")
    
    # Create vector store
    embedding_model = get_embedding_model(use_preset="fast")
    store = VectorStore(collection_name="test_rag", embedding_model=embedding_model)
    store.add_chunks(chunks)
    
    # Test retrieval
    test_queries = [
        "What medications should I take?",
        "When should I go to the emergency room?",
        "What caused my chest pain?",
    ]
    
    print("\n" + "-"*40)
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = store.search(query, n_results=1)
        if results:
            print(f"   Score: {results[0].score:.3f}")
            print(f"   Section: {results[0].metadata.get('section', 'unknown')}")
            preview = results[0].content[:100].replace('\n', ' ')
            print(f"   Content: {preview}...")
    
    return store


def test_full_rag_pipeline():
    """
    Full RAG test with LLM simplification.
    
    NOTE: This downloads and loads an LLM (~2GB for Phi-3)
    """
    print("\n" + "="*60)
    print("TESTING FULL RAG + SIMPLIFICATION PIPELINE")
    print("="*60)
    print("\n⚠️  This will download the LLM on first run (~2GB)")
    print("    Press Ctrl+C to skip if you want to test without LLM")
    
    try:
        input("\nPress Enter to continue or Ctrl+C to skip...")
    except KeyboardInterrupt:
        print("\n\nSkipped LLM test.")
        return None
    
    # Import here to avoid loading LLM unless needed
    from app.generation.simplifier import create_simplifier
    
    # Set up retrieval
    doc = create_sample_document()
    chunker = MedicalTextChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)
    
    embedding_model = get_embedding_model(use_preset="fast")
    store = VectorStore(collection_name="test_full", embedding_model=embedding_model)
    store.add_chunks(chunks)
    
    # Load simplifier (Phi-3 is smallest, ~2GB)
    print("\nLoading LLM...")
    simplifier = create_simplifier(use_preset="phi3")
    
    # Set up readability scorer
    scorer = ReadabilityScorer()
    
    # Test RAG + Simplification
    print("\n" + "-"*40)
    print("TESTING RAG-POWERED Q&A")
    print("-"*40)
    
    questions = [
        "What medications do I need to take and how often?",
        "What warning signs should make me go back to the hospital?",
    ]
    
    for question in questions:
        print(f"\n❓ Patient Question: {question}")
        
        # Retrieve relevant context
        results = store.search(question, n_results=2)
        context = "\n\n".join([r.content for r in results])
        
        # Generate answer
        print("   Generating answer...")
        answer = simplifier.answer_question(question, context)
        
        print(f"\n💬 Answer:\n{answer}")
        
        # Check readability
        readability = scorer.score(answer)
        print(f"\n📊 Readability: Grade {readability.avg_grade_level:.1f} ", end="")
        print("✓" if readability.is_patient_friendly else "✗")
    
    return simplifier, store, scorer


def main():
    """Run tests progressively."""
    print("\n" + "🏥 PATIENT COMMUNICATION ASSISTANT - FULL PIPELINE TEST 🏥")
    
    # Test 1: Readability (no dependencies)
    test_readability_only()
    
    # Test 2: Retrieval (needs embeddings model ~90MB)
    test_retrieval_pipeline()
    
    # Test 3: Full pipeline with LLM (needs ~2GB+ model)
    test_full_rag_pipeline()
    
    print("\n" + "="*60)
    print("✅ TESTING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()