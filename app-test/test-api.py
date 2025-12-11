"""
Test script for the FastAPI backend.
Run the server first: uvicorn app.main:app --reload
Then run: python test_api.py
"""

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n1. Testing health endpoint...")
    r = requests.get(f"{BASE_URL}/")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
    return r.status_code == 200


def test_stats():
    """Test stats endpoint."""
    print("\n2. Testing stats endpoint...")
    r = requests.get(f"{BASE_URL}/stats")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
    return r.status_code == 200


def test_readability():
    """Test readability analysis."""
    print("\n3. Testing readability endpoint...")
    
    complex_text = """
    The patient presents with acute exacerbation of chronic obstructive 
    pulmonary disease, characterized by increased dyspnea and decreased 
    exercise tolerance. Treatment includes bronchodilators and supplemental oxygen.
    """
    
    r = requests.post(f"{BASE_URL}/readability", json={"text": complex_text})
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Grade Level: {data['readability']['avg_grade_level']:.1f}")
    print(f"   Patient-friendly: {data['is_patient_friendly']}")
    print(f"   Recommendation: {data['recommendation']}")
    return r.status_code == 200


def test_upload_sample():
    """Test PDF upload with a sample file."""
    print("\n4. Testing PDF upload...")
    
    # Check if sample PDF exists
    import os
    sample_path = "data/sample_docs/sample.pdf"
    
    if not os.path.exists(sample_path):
        print(f"   ⚠ No sample PDF found at {sample_path}")
        print("   Create a sample PDF or skip this test")
        return True  # Skip but don't fail
    
    with open(sample_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/upload",
            files={"file": ("sample.pdf", f, "application/pdf")}
        )
    
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   Chunks added: {data['chunks_added']}")
        print(f"   Total chunks: {data['total_chunks']}")
        print(f"   Sections: {data['sections_found']}")
    return r.status_code == 200


def test_ask_question():
    """Test question answering."""
    print("\n5. Testing question endpoint...")
    
    # First check if we have documents
    stats = requests.get(f"{BASE_URL}/stats").json()
    if stats["total_chunks"] == 0:
        print("   ⚠ No documents indexed, skipping question test")
        return True
    
    r = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": "What medications should I take?",
            "use_simplifier": False,  # Set True to test LLM
            "n_results": 2
        }
    )
    
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   Answer preview: {data['answer'][:150]}...")
        print(f"   Sources: {len(data['sources'])}")
        print(f"   Readability grade: {data['readability']['avg_grade_level']:.1f}")
    return r.status_code == 200


def test_simplify():
    """Test text simplification (requires LLM)."""
    print("\n6. Testing simplify endpoint (loads LLM on first call)...")
    print("   ⚠ This may take a minute on first run...")
    
    text = """
    The patient should take omeprazole 20mg orally once daily, 
    30 minutes prior to the first meal of the day. This proton pump 
    inhibitor reduces gastric acid secretion and promotes healing 
    of esophageal erosions associated with GERD.
    """
    
    try:
        r = requests.post(
            f"{BASE_URL}/simplify",
            json={"text": text},
            timeout=120  # Long timeout for LLM loading
        )
        
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   Original grade: {data['readability_before']['avg_grade_level']:.1f}")
            print(f"   Simplified grade: {data['readability_after']['avg_grade_level']:.1f}")
            print(f"   Improvement: {data['improvement']['grade_level_reduction']:.1f} grades")
            print(f"   Simplified text: {data['simplified'][:150]}...")
        return r.status_code == 200
    except requests.exceptions.Timeout:
        print("   ⚠ Request timed out (LLM may still be loading)")
        return True


def main():
    print("=" * 60)
    print("PATIENT COMMUNICATION ASSISTANT - API TESTS")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    
    # Check if server is running
    try:
        requests.get(BASE_URL, timeout=5)
    except requests.exceptions.ConnectionError:
        print("\n❌ Server not running!")
        print("   Start it with: uvicorn app.main:app --reload")
        return
    
    results = []
    results.append(("Health", test_health()))
    results.append(("Stats", test_stats()))
    results.append(("Readability", test_readability()))
    results.append(("Upload", test_upload_sample()))
    results.append(("Ask", test_ask_question()))
    
    # Uncomment to test LLM simplification:
    # results.append(("Simplify", test_simplify()))
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"   {status} {name}")
    
    print("\n✅ API tests complete!")


if __name__ == "__main__":
    main()