"""
FastAPI backend for Patient Communication Assistant.
Run: uvicorn app.main:app --reload
"""

import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tempfile

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.ingestion.pdf_loader import MedicalPDFLoader
from app.ingestion.chunker import MedicalTextChunker
from app.retrieval.embeddings import get_embedding_model
from app.retrieval.vector_store import VectorStore
from app.evaluation.readability import ReadabilityScorer
from app.generation.errors import InputTooLongError, LLMRequestError, LLMUnavailableError
from app.generation.providers import LLMNotConfiguredError, build_provider_chain
from app.ratelimit import LLMRateLimiter, RateLimited, client_id
from app.privacy import Deidentifier, find_names


# ============================================================
# Global State (initialized on startup)
# ============================================================

class AppState:
    vector_store: Optional[VectorStore] = None
    embedding_model = None
    simplifier = None  # Provider chain; models load on first use
    llm_error: Optional[str] = None  # why no LLM is available, if so
    rate_limiter = LLMRateLimiter(
        per_client_per_minute=settings.llm_requests_per_minute,
        per_day=settings.llm_requests_per_day,
    )
    readability_scorer = ReadabilityScorer(target_grade_level=settings.target_grade_level)
    chunker = MedicalTextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
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
    state.embedding_model = get_embedding_model(
        use_preset=settings.embedding_preset,
        backend=settings.embedding_backend,
        cache_dir=settings.model_cache_dir,
    )

    # Create vector store
    state.vector_store = VectorStore(
        collection_name=settings.collection_name,
        persist_directory=settings.vector_store_path,
        embedding_model=state.embedding_model
    )
    print(f"✓ Vector store ready ({state.vector_store.get_chunk_count()} chunks)")

    # LLM providers (nothing is loaded or called yet)
    if settings.enable_llm:
        try:
            state.simplifier = build_provider_chain(settings)
            state.llm_error = None
            print(f"✓ LLM: {' → '.join(state.simplifier.labels)}")
        except LLMNotConfiguredError as e:
            state.simplifier, state.llm_error = None, str(e)
            print(f"⚠ {e}")
    
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
    question: str = Field(max_length=2000)
    use_simplifier: bool = False  # Set to True to use LLM simplification
    n_results: int = Field(default=3, ge=1, le=10)

class SimplifyRequest(BaseModel):
    text: str

class ReadabilityRequest(BaseModel):
    text: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    readability: dict
    identifiers_removed: int = 0  # replaced before the text reached the LLM

class SimplifyResponse(BaseModel):
    original: str
    simplified: str
    model_used: Optional[str] = None
    readability_before: dict
    readability_after: dict
    improvement: dict
    identifiers_removed: int = 0  # replaced before the text reached the LLM

class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int
    sections_found: list[str]


# ============================================================
# Helper Functions
# ============================================================

LLM_OFF = "AI simplification is not available on this deployment."


def get_simplifier(request: Request):
    """The LLM provider chain, after checking it's enabled and rate limits."""
    if not settings.enable_llm or state.simplifier is None:
        if settings.enable_llm and state.llm_error:
            print(f"⚠ LLM request refused: {state.llm_error}")
        raise HTTPException(status_code=503, detail=LLM_OFF)
    try:
        state.rate_limiter.check(client_id(request))
    except RateLimited as e:
        raise HTTPException(status_code=429, detail=str(e),
                            headers={"Retry-After": str(e.retry_after)})
    return state.simplifier


def run_llm(fn, *args):
    """Call a simplifier method, mapping provider errors to HTTP errors.

    Details are logged, not returned, so provider messages stay server-side.
    """
    try:
        return fn(*args)
    except InputTooLongError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except LLMUnavailableError as e:
        print(f"⚠ LLM unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="AI simplification is temporarily unavailable. Please try again in a minute.",
        )
    except LLMRequestError as e:
        print(f"✗ LLM request rejected: {e}")
        raise HTTPException(status_code=502, detail="The AI provider rejected the request.")


class _NoDeidentification:
    """Stand-in used when DEIDENTIFY=false."""
    count = 0

    def redact(self, text: str) -> str:
        return text

    def restore(self, text: str) -> str:
        return text


def new_deidentifier(known_names=()):
    """One per request, so a value keeps the same placeholder throughout."""
    if not settings.deidentify:
        return _NoDeidentification()
    return Deidentifier(known_names=list(known_names))


PHI_NAMES_KEY = "phi_names"


def names_from_results(results) -> list[str]:
    """Names found in each source document's header at upload time."""
    names = []
    for r in results:
        stored = r.metadata.get(PHI_NAMES_KEY)
        if stored:
            names.extend(stored.split("\n"))
    return list(dict.fromkeys(names))


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
        "chunks_indexed": state.vector_store.get_chunk_count() if state.vector_store else 0,
        "llm_enabled": settings.enable_llm and state.simplifier is not None,
        "llm_providers": getattr(state.simplifier, "labels", []) if settings.enable_llm else [],
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
        
        # Names from the document's header ("Patient: ..."), kept with every
        # chunk so they can be removed from chunks that don't include it
        names = find_names(doc.content)

        # Update source file name to original
        for chunk in chunks:
            chunk.source_file = file.filename
            if names:
                chunk.metadata[PHI_NAMES_KEY] = "\n".join(names)
        
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


# Plain `def` (not async): FastAPI runs these in a threadpool, so a slow LLM
# call doesn't block other requests
@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest, http_request: Request):
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
    identifiers_removed = 0
    if request.use_simplifier and results:
        # Use LLM to generate simplified answer, de-identified on the way in
        # and re-identified on the way out
        simplifier = get_simplifier(http_request)
        deid = new_deidentifier(names_from_results(results))
        question = deid.redact(request.question)
        context = deid.redact("\n\n".join([r.content for r in results]))
        answer = deid.restore(run_llm(simplifier.answer_question, question, context))
        identifiers_removed = deid.count
    else:
        # Return formatted context directly
        answer = format_context_for_answer(results)
    
    # Check readability
    readability = state.readability_scorer.score(answer)
    
    return QuestionResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        readability=readability.to_dict(),
        identifiers_removed=identifiers_removed,
    )


@app.post("/simplify", response_model=SimplifyResponse)
def simplify_text(request: SimplifyRequest, http_request: Request):
    """Simplify medical text to patient-friendly language."""
    if len(request.text) > settings.max_input_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Text is {len(request.text)} characters; the limit is {settings.max_input_chars}.",
        )
    simplifier = get_simplifier(http_request)
    
    # Get readability before
    readability_before = state.readability_scorer.score(request.text)
    
    # Simplify, with identifiers replaced before the text reaches the LLM
    deid = new_deidentifier()
    result = run_llm(simplifier.simplify_text, deid.redact(request.text))
    simplified = deid.restore(result.simplified_text)

    # Get readability after
    readability_after = state.readability_scorer.score(simplified)
    
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
        simplified=simplified,
        model_used=getattr(result, "model_used", None),
        readability_before=readability_before.to_dict(),
        readability_after=readability_after.to_dict(),
        improvement=improvement,
        identifiers_removed=deid.count,
    )


@app.post("/readability")
async def analyze_readability(request: ReadabilityRequest):
    """Analyze readability of text without simplification."""
    score = state.readability_scorer.score(request.text)
    return {
        "text_preview": request.text[:200] + "..." if len(request.text) > 200 else request.text,
        "readability": score.to_dict(),
        "is_patient_friendly": score.is_patient_friendly,
        "recommendation": (
            "This text is patient-friendly!" 
            if score.is_patient_friendly 
            else f"Consider simplifying. Current grade level: {score.avg_grade_level:.1f}, target: ≤{settings.target_grade_level:g}"
        )
    }


@app.delete("/clear")
async def clear_index():
    """Clear all indexed documents."""
    state.vector_store.clear()
    return {"status": "cleared", "chunks_remaining": 0}