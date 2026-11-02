"""Open-source embedding models.

Two backends:
- "onnx": all-MiniLM-L6-v2 through Chroma's bundled ONNX runtime. No PyTorch
  needed, so it fits small hosts (e.g. 512MB). Produces the same vectors as
  sentence-transformers for this model.
- "sentence-transformers": any HuggingFace model, via PyTorch.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np


ONNX_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EmbeddingResult:
    """Container for embedding results."""
    text: str
    embedding: list[float]
    model: str
    dimensions: int


def _onnx_embedder(cache_dir: Optional[str] = None):
    """Chroma's ONNX MiniLM, optionally downloading to/reading from cache_dir."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    embedder = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
    if cache_dir:
        embedder.DOWNLOAD_PATH = Path(cache_dir) / "onnx" / ONNXMiniLM_L6_V2.MODEL_NAME
    return embedder


class EmbeddingModel:
    """
    Wrapper for embedding models.

    Recommended models:
    - General: "sentence-transformers/all-MiniLM-L6-v2" (fast, 384 dims; ONNX)
    - General (better): "sentence-transformers/all-mpnet-base-v2" (768 dims)
    - Medical: "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
    - Medical: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    """

    # Model presets for easy selection
    MODEL_PRESETS = {
        "fast": ONNX_MODEL,
        "balanced": "sentence-transformers/all-mpnet-base-v2",
        "medical": "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "medical-alt": "NeuML/pubmedbert-base-embeddings",
    }

    def __init__(
        self,
        model_name: str = ONNX_MODEL,
        device: Optional[str] = None,
        use_preset: Optional[str] = None,
        backend: str = "auto",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model name or path
            device: Torch device ('cpu', 'cuda', 'mps'); sentence-transformers only.
            use_preset: Use a preset model ('fast', 'balanced', 'medical')
            backend: "auto" (ONNX for all-MiniLM-L6-v2, else sentence-transformers),
                "onnx" or "sentence-transformers"
            cache_dir: Where the ONNX model is downloaded (default ~/.cache/chroma)
        """
        if use_preset and use_preset in self.MODEL_PRESETS:
            model_name = self.MODEL_PRESETS[use_preset]

        if backend == "auto":
            backend = "onnx" if model_name == ONNX_MODEL else "sentence-transformers"
        if backend == "onnx" and model_name != ONNX_MODEL:
            raise ValueError(f"The ONNX backend only supports {ONNX_MODEL}")
        if backend not in ("onnx", "sentence-transformers"):
            raise ValueError(f"Unknown embedding backend {backend!r}")

        self.model_name = model_name
        self.backend = backend

        if backend == "onnx":
            self._onnx = _onnx_embedder(cache_dir)
            self._onnx(["warm up"])  # download if needed and load the session now
            self.dimensions = 384
            device_label = "cpu (onnxruntime)"
        else:
            # Imported lazily so the ONNX path never loads PyTorch
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self.dimensions = self.model.get_sentence_embedding_dimension()
            device_label = str(self.model.device)

        print(f"✓ Loaded embedding model: {model_name} [{backend}]")
        print(f"  Dimensions: {self.dimensions}")
        print(f"  Device: {device_label}")

    def _encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        if self.backend == "onnx":
            return np.asarray(self._onnx(texts), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        embedding = self._encode([text])[0]

        return EmbeddingResult(
            text=text,
            embedding=embedding.tolist(),
            model=self.model_name,
            dimensions=self.dimensions
        )

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar (sentence-transformers only)
        """
        embeddings = self._encode(texts, batch_size=batch_size, show_progress=show_progress)

        return [
            EmbeddingResult(
                text=text,
                embedding=embeddings[i].tolist(),
                model=self.model_name,
                dimensions=self.dimensions
            )
            for i, text in enumerate(texts)
        ]

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query (convenience method)."""
        result = self.embed_text(query)
        return result.embedding

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = np.array(self.embed_query(text1))
        emb2 = np.array(self.embed_query(text2))
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def get_embedding_model(
    model_name: str = ONNX_MODEL,
    use_preset: Optional[str] = None,
    backend: str = "auto",
    cache_dir: Optional[str] = None,
) -> EmbeddingModel:
    """
    Factory function to get an embedding model.

    Args:
        model_name: HuggingFace model name
        use_preset: Use preset ('fast', 'balanced', 'medical')
        backend: 'auto', 'onnx' or 'sentence-transformers'
        cache_dir: ONNX model download directory

    Examples:
        # Fast general-purpose model (ONNX, no PyTorch)
        model = get_embedding_model(use_preset="fast")

        # Medical domain model (sentence-transformers)
        model = get_embedding_model(use_preset="medical")

        # Custom model
        model = get_embedding_model("your-org/your-model")
    """
    return EmbeddingModel(
        model_name=model_name, use_preset=use_preset, backend=backend, cache_dir=cache_dir
    )
