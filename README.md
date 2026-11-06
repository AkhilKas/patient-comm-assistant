# 🏥 Patient Communication Assistant

An AI-powered RAG (Retrieval-Augmented Generation) application that transforms complex medical documents into patient-friendly explanations at an 8th-grade reading level.

[![CI](https://github.com/AkhilKas/patient-comm-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/AkhilKas/patient-comm-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![React](https://img.shields.io/badge/React-18.2-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-yellow.svg)

**🔗 Live demo: [patient-comm-assistant.onrender.com](https://patient-comm-assistant.onrender.com)**

> The demo runs on Render's free tier. The first request after it has been idle can take about a minute while the backend wakes up. Uploaded documents are cleared when it sleeps. AI simplification sends text to Google's Gemini API after names, dates and contact details have been replaced, and Google may use free-tier requests to improve its products. Please use sample or de-identified documents only, not real patient records.

## 🎯 Problem Statement

Patients often struggle to understand complex medical discharge instructions, medication guides, and doctor notes. Medical jargon and technical language create barriers to proper self-care, leading to:
- Medication non-adherence
- Missed follow-up appointments
- Preventable hospital readmissions
- Poor health outcomes

## 💡 Solution

A two-stage RAG pipeline that:
1. **Ingests** medical PDFs with intelligent chunking
2. **Retrieves** relevant information using semantic search
3. **Simplifies** complex text to an 8th-grade reading level
4. **Validates** output with automated readability scoring

## ✨ Features

- **PDF Document Ingestion**: Upload discharge summaries, medication guides, and clinical notes
- **Intelligent Chunking**: Section-aware text splitting optimized for medical documents
- **Semantic Search**: Vector similarity search using Sentence Transformers embeddings
- **AI Simplification**: Two-stage LLM pipeline for text simplification
- **De-identification**: Names, dates, phone numbers, IDs and addresses are replaced with placeholders before any text reaches the LLM, then restored in the answer
- **Readability Scoring**: Automated Flesch-Kincaid, Gunning Fog, SMOG, and Coleman-Liau analysis
- **Interactive Q&A**: Ask questions about uploaded documents in natural language
- **Modern Web Interface**: Responsive React frontend with real-time feedback

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PDF Upload    │────▶│   Ingestion     │────▶│  Vector Store   │
│                 │     │  (PyPDF/Plumber)│     │   (ChromaDB)    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────▼────────┐
│  Simplified     │◀────│   Generation    │◀────│    Retrieval    │
│   Response      │     │ (Gemini/Phi-3)  │     │ (Semantic Search│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                        ┌───────▼───────┐
                        │  Readability  │
                        │   Validation  │
                        └───────────────┘
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** - High-performance async API framework
- **ChromaDB** - Vector database for semantic search
- **ONNX Runtime / Sentence Transformers** - Open-source embeddings (all-MiniLM-L6-v2 runs on ONNX, no PyTorch needed; other presets use Sentence Transformers)
- **Gemini API** - Default LLM when `GEMINI_API_KEY` is set (plain REST via httpx, no SDK)
- **Hugging Face Transformers** - Local LLM inference (Phi-3, Mistral, Llama) on CUDA / MPS / CPU, used as the fallback or on its own
- **MLX (mlx-lm)** - 4-bit LLM inference on Apple Silicon, used automatically when installed (`LLM_BACKEND=auto`)
- **PyPDF2 / pdfplumber** - PDF text extraction
- **textstat** - Readability metrics calculation

### Frontend
- **React 18** - Component-based UI
- **Tailwind CSS** - Utility-first styling
- **Vite** - Fast build tooling
- **Lucide React** - Icon library

## 📁 Project Structure

```
patient-comm-assistant/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Settings (env vars / .env, see .env.example)
│   ├── ingestion/
│   │   ├── pdf_loader.py       # PDF extraction & preprocessing
│   │   └── chunker.py          # Medical-aware text chunking
│   ├── retrieval/
│   │   ├── embeddings.py       # Sentence Transformer wrapper
│   │   └── vector_store.py     # ChromaDB operations
│   ├── generation/
│   │   ├── simplifier.py       # LLM simplification pipeline
│   │   └── prompts.py          # Prompt templates
│   ├── privacy/
│   │   └── deid.py             # De-identification before the LLM
│   └── evaluation/
│       └── readability.py      # Readability scoring
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main React component
│   │   ├── main.jsx            # Entry point
│   │   └── index.css           # Tailwind imports
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── tests/                      # pytest suite
├── scripts/check_memory.py     # Peak-memory check used by CI
├── render.yaml                 # Render deployment Blueprint
├── requirements-core.txt       # Lean API server (no PyTorch)
├── requirements.txt            # Full: core + local LLMs + dev tools
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ (required by transformers 5)
- Node.js 18+
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey)), **or** 8GB+ RAM to run the local LLM (4GB+ with the 4-bit MLX models on Apple Silicon)

### Backend Setup

```bash
# Clone repository
git clone https://github.com/AkhilKas/patient-comm-assistant.git
cd patient-comm-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (full: includes PyTorch and local LLM support)
pip install -r requirements.txt
# ...or the lean server only (no PyTorch, no LLM features):
# pip install -r requirements-core.txt

# Optional: override defaults (model presets, paths, HF token)
cp .env.example .env

# Start backend server
uvicorn app.main:app --reload
```

### Running Tests

```bash
pytest                      # ingestion, readability, retrieval, API
RUN_LLM_TESTS=1 pytest      # also run LLM tests (downloads the model)
```

No running server is needed; the API tests use FastAPI's `TestClient` with a temporary vector store.

GitHub Actions (`.github/workflows/ci.yml`) runs the test suite on Python 3.11 and 3.12 and builds the frontend on every pull request and every push to `main`.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Access Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🤖 LLM Providers

| `LLM_PROVIDER` | Behaviour |
|---|---|
| `auto` (default) | Gemini if `GEMINI_API_KEY` is set, with the local model as fallback when it's installed. Otherwise the local model. |
| `gemini` | Gemini only (plus the local fallback if `LLM_FALLBACK=true` and it's installed). |
| `local` | The local transformers/MLX model only. Nothing leaves your machine. |

- **Fallback:** when Gemini times out, is rate limited (429), has a server error (5xx) or returns no answer, the request is retried on the local model, which loads on first use. A rejected request (bad key, unknown model) is not retried, so configuration mistakes aren't hidden. `/simplify` reports which model answered in `model_used`.
- **No local model on Render:** the lean install has no PyTorch, so there a Gemini failure returns 503.
- **Model:** `GEMINI_MODEL`, default `gemini-3.5-flash-lite`. The key is sent in a header, never in the URL.
- **Quota protection:** LLM requests are limited to `LLM_REQUESTS_PER_MINUTE` per client (default 10) and `LLM_REQUESTS_PER_DAY` overall (default 500), returning 429 with `Retry-After`. `/simplify` input is capped at `MAX_INPUT_CHARS` (20,000).
- **Privacy:** text is de-identified before it reaches any provider (next section). On Gemini's free tier, Google may use prompts and responses to improve its products. Use `LLM_PROVIDER=local` for anything sensitive.

## 🔒 De-identification

Before any text is sent to the LLM (`/simplify`, and `/ask` with `use_simplifier`), `app/privacy/deid.py` replaces patient identifiers with numbered placeholders. The real values are put back into the answer, so the patient still sees "Call Dr. Patel at (413) 555-0147", but the model only ever sees "Call [NAME_1] at [PHONE_1]".

```
Patient: [NAME_1]    DOB: [DATE_1]    MRN: [ID_1]
[NAME_2] is a 67-year-old man admitted with chest pain. See [NAME_3] on [DATE_2].
```

- **What it catches:** labelled fields (`Patient:`, `DOB:`, `MRN:`, `Address:`, `Attending:`, ...), emails, URLs, phone numbers, SSNs, dates, street addresses, ZIP codes, long ID numbers, ages over 89, and titled names ("Dr. Patel", "Jane Doe, MD"). Names from a document's header are saved at upload, so they are also removed from chunks that don't contain the header.
- **How it works:** rules only, with no model, so it adds nothing to the backend's memory. The placeholder mapping lives in memory for the one request and is never stored or sent.
- **Limits:** a name with no label, title or earlier mention (e.g. "your daughter Maria") is not detected. This reduces what reaches the LLM but is not a guarantee, and the app is not HIPAA compliant. Use sample or de-identified documents.
- **Check a document:** `python -m app.privacy note.pdf` prints the text as the LLM would see it, followed by the list of replacements.
- **Turn it off:** `DEIDENTIFY=false` (not recommended with a hosted LLM).
- Tests (`tests/test_deid.py`, `tests/test_api.py`) fail if any identifier from a synthetic note reaches a recording fake LLM.

## ☁️ Deployment (Render)

`render.yaml` is a [Render Blueprint](https://render.com/docs/blueprint-spec) that deploys both services. In the Render dashboard, choose **New → Blueprint** and select this repo.

Both services run on Render's **free** tier:

| Service | Type | Plan | Notes |
|---|---|---|---|
| `patient-comm-assistant` | Static site | Free | **The app:** [patient-comm-assistant.onrender.com](https://patient-comm-assistant.onrender.com). Rewrites `/api/*` to the backend. |
| `patient-comm-api` | Python web service | Free (512MB) | **API only:** [patient-comm-api.onrender.com](https://patient-comm-api.onrender.com) returns the health check JSON. Installs `requirements-core.txt` only; models downloaded at build time. |

- **Lean backend:** no PyTorch. Embeddings run on ONNX (same vectors as Sentence Transformers). A CI job installs only the core requirements and fails if peak memory goes over 400MB.
- **LLM in production:** Gemini. Add `GEMINI_API_KEY` in the Render dashboard (the Blueprint prompts for it; it's never stored in `render.yaml`). Without a key, upload, Q&A retrieval and readability still work, and `/simplify` and simplified answers return 503.
- **Free-tier trade-offs:** the backend sleeps after 15 minutes idle, and the first request after that takes about a minute. There's no persistent disk, so **uploaded documents are cleared whenever the backend sleeps or restarts**.
- **Deploys:** both services deploy automatically from `main` once CI passes.
- **Backend URL:** the frontend's `/api/*` rewrite assumes the backend is at `https://patient-comm-api.onrender.com`. If Render assigns a different URL, update the rewrite in `render.yaml`, or set `VITE_API_URL` on the static site.

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/stats` | Index statistics |
| POST | `/upload` | Upload PDF document |
| POST | `/ask` | Ask question (RAG) |
| POST | `/simplify` | Simplify text (LLM) |
| POST | `/readability` | Analyze readability |
| DELETE | `/clear` | Clear all documents |

## 📈 Performance Metrics

- **Readability Improvement**: Average 12+ grade level reduction
- **Flesch Ease Improvement**: +60-80 points (from negative to 70+)
- **Embedding Latency**: <100ms for query embedding
- **Retrieval Latency**: <50ms for top-k search

## 🔮 Future Enhancements

- [ ] PubMedBERT embeddings for improved medical domain relevance
- [ ] Multi-language support for diverse patient populations
- [ ] Speech-to-text for accessibility
- [ ] Integration with EHR systems (FHIR API)
- [ ] Confidence scoring for generated responses

## 📄 License

MIT License - feel free to use for personal or commercial projects.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.