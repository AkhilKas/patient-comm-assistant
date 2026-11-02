"""Download models ahead of time (e.g. during a deploy build).

Run: python -m app.warmup

Fetches the ONNX embedding model into MODEL_CACHE_DIR and the tiktoken
encoding into TIKTOKEN_CACHE_DIR, so the server doesn't download them on
every cold start.
"""

from app.config import settings
from app.ingestion.chunker import MedicalTextChunker
from app.retrieval.embeddings import get_embedding_model


def main() -> None:
    get_embedding_model(
        use_preset=settings.embedding_preset,
        backend=settings.embedding_backend,
        cache_dir=settings.model_cache_dir,
    )
    MedicalTextChunker().count_tokens("warm up")
    print("✓ Models cached")


if __name__ == "__main__":
    main()
