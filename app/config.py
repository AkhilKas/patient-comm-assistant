"""Application configuration and settings.

Every field can be overridden by an environment variable of the same name
(case-insensitive), e.g. LLM_PRESET=mistral, or via a .env file.
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # HuggingFace token (only needed for gated models such as Llama 3)
    hf_token: Optional[str] = None

    # Embeddings: preset name from EmbeddingModel.MODEL_PRESETS
    embedding_preset: str = "fast"
    # "auto" uses ONNX (no PyTorch) for the "fast" preset, else sentence-transformers
    embedding_backend: str = "auto"
    # Where the ONNX embedding model is stored (default: ~/.cache/chroma)
    model_cache_dir: Optional[str] = None

    # Generation: set false to turn AI simplification off entirely;
    # /simplify and simplified /ask answers then return 503
    enable_llm: bool = True
    # "auto": Gemini if GEMINI_API_KEY is set (local model as fallback when
    # installed), else the local model. "gemini" or "local" to force one.
    llm_provider: str = "auto"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_timeout: float = 30.0
    # Fall back to the local model when Gemini times out, is rate limited or down
    llm_fallback: bool = True
    # Abuse/quota protection for LLM endpoints (0 = no limit)
    llm_requests_per_minute: int = 10  # per client
    llm_requests_per_day: int = 500  # all clients combined
    max_input_chars: int = 20000
    # Preset name from MedicalSimplifier.MODEL_PRESETS
    llm_preset: str = "phi3"
    # "auto" uses MLX on Apple Silicon when mlx-lm is installed, else torch
    llm_backend: str = "auto"
    load_in_4bit: bool = False  # torch backend on CUDA only, requires bitsandbytes
    # Replace names, dates, phone numbers, IDs, ... with placeholders before
    # any text is sent to the LLM, and put them back in the answer
    deidentify: bool = True

    # Vector store
    vector_store_path: str = "data/vector_store"
    collection_name: str = "patient_docs"

    # Chunking (token counts)
    chunk_size: int = 300
    chunk_overlap: int = 50

    # Readability
    target_grade_level: float = 8.0


settings = Settings()
