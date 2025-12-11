"""ChromaDB vector store for medical document retrieval."""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import chromadb

from .embeddings import EmbeddingModel, get_embedding_model

import sys
sys.path.append(str(Path(__file__).parent.parent))
from ingestion.chunker import TextChunk


@dataclass
class SearchResult:
    """Container for search results."""
    chunk_id: str
    content: str
    score: float
    metadata: dict


class VectorStore:
    """ChromaDB-based vector store for medical documents."""
    
    def __init__(
        self,
        collection_name: str = "discharge_docs",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        use_medical_embeddings: bool = False
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Path to persist data (None = in-memory)
            embedding_model: Custom embedding model (auto-creates if None)
            use_medical_embeddings: Use medical domain embeddings
        """
        self.collection_name = collection_name
        
        # Initialize embedding model
        if embedding_model:
            self.embedding_model = embedding_model
        elif use_medical_embeddings:
            self.embedding_model = get_embedding_model(use_preset="medical")
        else:
            self.embedding_model = get_embedding_model(use_preset="fast")
        
        # Initialize ChromaDB
        if persist_directory:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_path))
        else:
            self.client = chromadb.Client()
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: list[TextChunk], show_progress: bool = True) -> int:
        """Add text chunks to the vector store."""
        if not chunks:
            return 0
        
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source_file": chunk.source_file,
                "section": chunk.section or "unknown",
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]
        
        if show_progress:
            print(f"Generating embeddings for {len(chunks)} chunks...")
        
        embedding_results = self.embedding_model.embed_batch(documents, show_progress=show_progress)
        embeddings = [result.embedding for result in embedding_results]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        if show_progress:
            print(f"✓ Added {len(chunks)} chunks to vector store")
        
        return len(chunks)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_section: Optional[str] = None,
        filter_source: Optional[str] = None
    ) -> list[SearchResult]:
        """Search for similar chunks."""
        where_filter = None
        if filter_section or filter_source:
            conditions = []
            if filter_section:
                conditions.append({"section": filter_section})
            if filter_source:
                conditions.append({"source_file": filter_source})
            where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        
        query_embedding = self.embedding_model.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # cosine similarity
                
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    content=results["documents"][0][i],
                    score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                ))
        
        return search_results
    
    def search_by_section(self, query: str, section: str, n_results: int = 3) -> list[SearchResult]:
        """Search within a specific section."""
        return self.search(query, n_results=n_results, filter_section=section)
    
    def get_all_sections(self) -> list[str]:
        """Get all unique sections."""
        results = self.collection.get(include=["metadatas"])
        sections = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if meta and "section" in meta:
                    sections.add(meta["section"])
        return sorted(list(sections))
    
    def get_chunk_count(self) -> int:
        """Get total chunks in store."""
        return self.collection.count()
    
    def clear(self):
        """Clear all documents."""
        all_data = self.collection.get()
        if all_data["ids"]:
            self.collection.delete(ids=all_data["ids"])
            print(f"✓ Cleared {len(all_data['ids'])} chunks")


def create_vector_store(
    collection_name: str = "discharge_docs",
    persist_directory: Optional[str] = None,
    use_medical_embeddings: bool = False
) -> VectorStore:
    """
    Factory function to create a vector store.
    
    Args:
        collection_name: Name for the collection
        persist_directory: Path to persist (None = in-memory)
        use_medical_embeddings: Use PubMedBERT embeddings
    
    Examples:
        # Quick in-memory store
        store = create_vector_store()
        
        # Persistent store with medical embeddings
        store = create_vector_store(
            persist_directory="data/vector_store",
            use_medical_embeddings=True
        )
    """
    return VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        use_medical_embeddings=use_medical_embeddings
    )