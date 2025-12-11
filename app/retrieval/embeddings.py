"""Open-source embedding models via Sentence Transformers."""

from typing import Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np


@dataclass
class EmbeddingResult:
    """Container for embedding results."""
    text: str
    embedding: list[float]
    model: str
    dimensions: int


class EmbeddingModel:
    """
    Wrapper for Sentence Transformer embedding models.
    
    Recommended models:
    - General: "sentence-transformers/all-MiniLM-L6-v2" (fast, 384 dims)
    - General (better): "sentence-transformers/all-mpnet-base-v2" (768 dims)
    - Medical: "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
    - Medical: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    """
    
    # Model presets for easy selection
    MODEL_PRESETS = {
        "fast": "sentence-transformers/all-MiniLM-L6-v2",
        "balanced": "sentence-transformers/all-mpnet-base-v2",
        "medical": "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "medical-alt": "NeuML/pubmedbert-base-embeddings",
    }
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        use_preset: Optional[str] = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detects if None.
            use_preset: Use a preset model ('fast', 'balanced', 'medical')
        """
        if use_preset and use_preset in self.MODEL_PRESETS:
            model_name = self.MODEL_PRESETS[use_preset]
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimensions = self.model.get_sentence_embedding_dimension()
        
        print(f"✓ Loaded embedding model: {model_name}")
        print(f"  Dimensions: {self.dimensions}")
        print(f"  Device: {self.model.device}")
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        
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
            show_progress: Whether to show progress bar
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        results = []
        for i, text in enumerate(texts):
            results.append(EmbeddingResult(
                text=text,
                embedding=embeddings[i].tolist(),
                model=self.model_name,
                dimensions=self.dimensions
            ))
        
        return results
    
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
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_preset: Optional[str] = None
) -> EmbeddingModel:
    """
    Factory function to get an embedding model.
    
    Args:
        model_name: HuggingFace model name
        use_preset: Use preset ('fast', 'balanced', 'medical')
    
    Examples:
        # Fast general-purpose model
        model = get_embedding_model(use_preset="fast")
        
        # Medical domain model
        model = get_embedding_model(use_preset="medical")
        
        # Custom model
        model = get_embedding_model("your-org/your-model")
    """
    return EmbeddingModel(model_name=model_name, use_preset=use_preset)