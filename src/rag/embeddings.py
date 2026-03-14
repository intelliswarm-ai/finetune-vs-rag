"""
Embedding model for RAG pipeline
Model: sentence-transformers/all-MiniLM-L6-v2
"""
import time
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import model_config


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embeddings: np.ndarray
    latency_ms: float
    num_texts: int
    embedding_dim: int


class EmbeddingModel:
    """
    Embedding model for semantic search in RAG pipeline.
    Uses sentence-transformers for efficient text embedding.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or model_config.EMBEDDING_MODEL_ID
        self.model = None
        self._loaded = False
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension

    def load(self) -> None:
        """Load the embedding model"""
        if self._loaded:
            return

        print(f"Loading embedding model: {self.model_id}")
        self.model = SentenceTransformer(self.model_id)
        self._loaded = True
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded! Dimension: {self.embedding_dim}")

    def embed(self, texts: List[str], show_progress: bool = False) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            EmbeddingResult with embeddings and metrics
        """
        if not self._loaded:
            self.load()

        start_time = time.perf_counter()

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True  # For cosine similarity
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        return EmbeddingResult(
            embeddings=embeddings,
            latency_ms=latency_ms,
            num_texts=len(texts),
            embedding_dim=self.embedding_dim
        )

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text and return the embedding vector"""
        result = self.embed([text])
        return result.embeddings[0]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query for retrieval (alias for embed_single)"""
        return self.embed_single(query)

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if not self._loaded:
            self.load()
        return self.embedding_dim


# Singleton instance
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the embedding model instance"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
