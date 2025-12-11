"""
FastAPI backend for Patient Communication Assistant.
Run: uvicorn app.main:app --reload
"""

import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.pdf_loader import MedicalPDFLoader
from app.ingestion.chunker import MedicalTextChunker
from app.retrieval.embeddings import get_embedding_model
from app.retrieval.vector_store import VectorStore
from app.evaluation.readability import ReadabilityScorer, check_readability


# ============================================================
# Global State (initialized on startup)
# ============================================================

class AppState:
    vector_store: Optional[VectorStore] = None
    embedding_model = None
    simplifier = None  # Loaded lazily to save memory
    readability_scorer = ReadabilityScorer(target_grade_level=8.0)
    chunker = MedicalTextChunker(chunk_size=300, chunk_overlap=50)
    pdf_loader = MedicalPDFLoader()

state = AppState()


# ============================================================
# Lifespan (startup/shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Patient Communication Assistant...")
    
    # Load embedding model
    print("Loading embedding model...")
    state.embedding_model = get_embedding_model(use_preset="fast")
    
    # Create vector store
    persist_dir = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    state.vector_store = VectorStore(
        collection_name="patient_docs",
        persist_directory=persist_dir,
        embedding_model=state.embedding_model
    )
    print(f"✓ Vector store ready ({state.vector_store.get_chunk_count()} chunks)")
    
    print("✅ Backend ready!")
    
    yield
    
    # Shutdown
    print("Shutting down...")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Patient Communication Assistant",
    description="RAG-powered medical document simplification",
    version="0.1.0",
    lifespan=lifespan
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================

class QuestionRequest(BaseModel):
    question: str
    use_simplifier: bool = False  # Set to True to use LLM simplification
    n_results: int = 3

class SimplifyRequest(BaseModel):
    text: str

class ReadabilityRequest(BaseModel):
    text: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    readability: dict

class SimplifyResponse(BaseModel):
    original: str
    simplified: str
    readability_before: dict
    readability_after: dict
    improvement: dict

class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int
    sections_found: list[str]


# ============================================================
# Helper Functions
# ============================================================

def get_simplifier():
    """Lazy load the LLM simplifier to save memory."""
    if state.simplifier is None:
        print("Loading LLM simplifier (first request, may take a moment)...")
        from app.generation.simplifier import create_simplifier
        state.simplifier = create_simplifier(use_preset="phi3")
    return state.simplifier


def format_context_for_answer(results) -> str:
    """Format retrieved chunks into a readable answer."""
    if not results:
        return "I couldn't find relevant information in your documents."
    
    # Group by section
    sections = {}
    for r in results:
        section = r.metadata.get("section", "General")
        if section not in sections:
            sections[section] = []
        sections[section].append(r.content)
    
    # Format response
    parts = []
    for section, contents in sections.items():
        section_title = section.replace("_", " ").title()
        combined = " ".join(contents)
        parts.append(f"**{section_title}:**\n{combined}")
    
    return "\n\n".join(parts)


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Patient Communication Assistant",
        "chunks_indexed": state.vector_store.get_chunk_count() if state.vector_store else 0
    }


@app.get("/stats")
async def get_stats():
    """Get current index statistics."""
    return {
        "total_chunks": state.vector_store.get_chunk_count(),
        "sections": state.vector_store.get_all_sections(),
        "embedding_model": state.embedding_model.model_name if state.embedding_model else None,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a PDF document."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load and process PDF
        doc = state.pdf_loader.load(tmp_path)
        chunks = state.chunker.chunk_document(doc)
        
        # Update source file name to original
        for chunk in chunks:
            chunk.source_file = file.filename
        
        # Add to vector store
        state.vector_store.add_chunks(chunks, show_progress=False)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return UploadResponse(
            filename=file.filename,
            chunks_added=len(chunks),
            total_chunks=state.vector_store.get_chunk_count(),
            sections_found=doc.metadata.sections_found
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about uploaded documents.
    Returns relevant information with readability score.
    """
    if state.vector_store.get_chunk_count() == 0:
        raise HTTPException(
            status_code=400, 
            detail="No documents indexed. Please upload a document first."
        )
    
    # Retrieve relevant chunks
    results = state.vector_store.search(request.question, n_results=request.n_results)
    
    # Format sources
    sources = [
        {
            "content": r.content,
            "section": r.metadata.get("section", "unknown"),
            "source_file": r.metadata.get("source_file", "unknown"),
            "score": round(r.score, 3)
        }
        for r in results
    ]
    
    # Generate answer
    if request.use_simplifier and results:
        # Use LLM to generate simplified answer
        simplifier = get_simplifier()
        context = "\n\n".join([r.content for r in results])
        answer = simplifier.answer_question(request.question, context)
    else:
        # Return formatted context directly
        answer = format_context_for_answer(results)
    
    # Check readability
    readability = check_readability(answer)
    
    return QuestionResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        readability=readability.to_dict()
    )


@app.post("/simplify", response_model=SimplifyResponse)
async def simplify_text(request: SimplifyRequest):
    """Simplify medical text to patient-friendly language."""
    simplifier = get_simplifier()
    
    # Get readability before
    readability_before = check_readability(request.text)
    
    # Simplify
    result = simplifier.simplify_text(request.text)
    
    # Get readability after
    readability_after = check_readability(result.simplified_text)
    
    # Calculate improvement
    improvement = {
        "grade_level_reduction": round(
            readability_before.avg_grade_level - readability_after.avg_grade_level, 1
        ),
        "flesch_ease_improvement": round(
            readability_after.flesch_reading_ease - readability_before.flesch_reading_ease, 1
        ),
        "met_target": readability_after.is_patient_friendly
    }
    
    return SimplifyResponse(
        original=request.text,
        simplified=result.simplified_text,
        readability_before=readability_before.to_dict(),
        readability_after=readability_after.to_dict(),
        improvement=improvement
    )


@app.post("/readability")
async def analyze_readability(request: ReadabilityRequest):
    """Analyze readability of text without simplification."""
    score = check_readability(request.text)
    return {
        "text_preview": request.text[:200] + "..." if len(request.text) > 200 else request.text,
        "readability": score.to_dict(),
        "is_patient_friendly": score.is_patient_friendly,
        "recommendation": (
            "This text is patient-friendly!" 
            if score.is_patient_friendly 
            else f"Consider simplifying. Current grade level: {score.avg_grade_level:.1f}, target: ≤8"
        )
    }


@app.delete("/clear")
async def clear_index():
    """Clear all indexed documents."""
    state.vector_store.clear()
    return {"status": "cleared", "chunks_remaining": 0}