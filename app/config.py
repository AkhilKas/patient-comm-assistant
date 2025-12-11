"""Application configuration and settings."""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    vector_store_dir: Path = data_dir / "vector_store"
    
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    
    # Generation settings
    generation_model: str = "gpt-4o-mini"
    target_reading_level: int = 8
    
    # Vector store
    collection_name: str = "discharge_docs"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()