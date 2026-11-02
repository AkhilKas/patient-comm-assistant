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

    # Generation: set false on hosts without enough RAM for an LLM (~8GB);
    # /simplify and simplified /ask answers then return 503
    enable_llm: bool = True
    # Preset name from MedicalSimplifier.MODEL_PRESETS
    llm_preset: str = "phi3"
    # "auto" uses MLX on Apple Silicon when mlx-lm is installed, else torch
    llm_backend: str = "auto"
    load_in_4bit: bool = False  # torch backend on CUDA only, requires bitsandbytes

    # Vector store
    vector_store_path: str = "data/vector_store"
    collection_name: str = "patient_docs"

    # Chunking (token counts)
    chunk_size: int = 300
    chunk_overlap: int = 50

    # Readability
    target_grade_level: float = 8.0


settings = Settings()
