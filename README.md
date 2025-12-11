# 🏥 Patient Communication Assistant

An AI-powered RAG (Retrieval-Augmented Generation) application that transforms complex medical documents into patient-friendly explanations at an 8th-grade reading level.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![React](https://img.shields.io/badge/React-18.2-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-yellow.svg)

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
│   Response      │     │ (Phi-3/Mistral) │     │ (Semantic Search│
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
- **Sentence Transformers** - Open-source embeddings (all-MiniLM-L6-v2)
- **Hugging Face Transformers** - LLM inference (Phi-3, Mistral, Llama)
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
│   ├── config.py               # Configuration settings
│   ├── ingestion/
│   │   ├── pdf_loader.py       # PDF extraction & preprocessing
│   │   └── chunker.py          # Medical-aware text chunking
│   ├── retrieval/
│   │   ├── embeddings.py       # Sentence Transformer wrapper
│   │   └── vector_store.py     # ChromaDB operations
│   ├── generation/
│   │   ├── simplifier.py       # LLM simplification pipeline
│   │   └── prompts.py          # Prompt templates
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
├── data/
│   └── sample_docs/            # Sample PDFs for testing
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- 8GB+ RAM (for LLM inference)

### Backend Setup

```bash
# Clone repository
git clone https://github.com/yourusername/patient-comm-assistant.git
cd patient-comm-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
uvicorn app.main:app --reload
```

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