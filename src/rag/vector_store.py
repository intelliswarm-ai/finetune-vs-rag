"""
Vector Store using ChromaDB for RAG pipeline
"""
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from ..config import rag_config, DATA_DIR
from .embeddings import EmbeddingModel, get_embedding_model


@dataclass
class RetrievalResult:
    """Result from vector store retrieval"""
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[float]
    latency_ms: float
    num_results: int


class VectorStore:
    """
    Vector store for document retrieval using ChromaDB.
    Supports both persistent and in-memory storage.
    """

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: EmbeddingModel = None
    ):
        self.collection_name = collection_name or rag_config.COLLECTION_NAME
        self.persist_directory = persist_directory or rag_config.PERSIST_DIRECTORY
        self.embedding_model = embedding_model or get_embedding_model()

        self.client = None
        self.collection = None
        self._initialized = False

    def initialize(self, persist: bool = True) -> None:
        """Initialize the ChromaDB client and collection"""
        if self._initialized:
            return

        print(f"Initializing vector store: {self.collection_name}")

        if persist:
            # Create persist directory
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        self._initialized = True
        print(f"Vector store initialized! Documents: {self.collection.count()}")

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs

        Returns:
            Number of documents added
        """
        if not self._initialized:
            self.initialize()

        # Generate IDs if not provided
        if ids is None:
            current_count = self.collection.count()
            ids = [f"doc_{current_count + i}" for i in range(len(documents))]

        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        embedding_result = self.embedding_model.embed(documents, show_progress=True)

        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embedding_result.embeddings.tolist(),
            metadatas=metadatas or [{}] * len(documents),
            ids=ids
        )

        print(f"Added {len(documents)} documents. Total: {self.collection.count()}")
        return len(documents)

    def query(
        self,
        query_text: str,
        n_results: int = None,
        where: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Query the vector store for similar documents.

        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            RetrievalResult with documents and metrics
        """
        if not self._initialized:
            self.initialize()

        n_results = n_results or rag_config.TOP_K

        start_time = time.perf_counter()

        # Embed query
        query_embedding = self.embedding_model.embed_query(query_text)

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract results
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        return RetrievalResult(
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            latency_ms=latency_ms,
            num_results=len(documents)
        )

    def query_with_threshold(
        self,
        query_text: str,
        n_results: int = None,
        threshold: float = None
    ) -> RetrievalResult:
        """
        Query with similarity threshold filtering.

        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            threshold: Minimum similarity score (0-1, higher is more similar)

        Returns:
            RetrievalResult with filtered documents
        """
        threshold = threshold or rag_config.SIMILARITY_THRESHOLD
        result = self.query(query_text, n_results)

        # Filter by threshold (ChromaDB returns distances, lower is better)
        # For cosine distance: similarity = 1 - distance
        filtered_docs = []
        filtered_metas = []
        filtered_dists = []

        for doc, meta, dist in zip(result.documents, result.metadatas, result.distances):
            similarity = 1 - dist
            if similarity >= threshold:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_dists.append(dist)

        return RetrievalResult(
            documents=filtered_docs,
            metadatas=filtered_metas,
            distances=filtered_dists,
            latency_ms=result.latency_ms,
            num_results=len(filtered_docs)
        )

    def count(self) -> int:
        """Get the number of documents in the collection"""
        if not self._initialized:
            self.initialize()
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection"""
        if not self._initialized:
            self.initialize()

        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store cleared!")

    def is_initialized(self) -> bool:
        """Check if vector store is initialized"""
        return self._initialized


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
