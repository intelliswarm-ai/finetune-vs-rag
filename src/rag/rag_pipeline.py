"""
Complete RAG Pipeline combining retrieval and generation
"""
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .vector_store import VectorStore, get_vector_store
from .embeddings import EmbeddingModel, get_embedding_model
from ..models.base_model import BaseModel, get_base_model
from ..config import rag_config


@dataclass
class RAGResponse:
    """Complete response from RAG pipeline"""
    answer: str
    retrieved_docs: List[str]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    num_docs_retrieved: int
    tokens_generated: int
    model_name: str = "RAG (Mistral-7B)"


class RAGPipeline:
    """
    Complete RAG pipeline: Retrieve relevant documents, then generate answer.
    Uses Mistral-7B as the generation model.
    """

    def __init__(
        self,
        vector_store: VectorStore = None,
        base_model: BaseModel = None,
        top_k: int = None
    ):
        self.vector_store = vector_store or get_vector_store()
        self.base_model = base_model or get_base_model()
        self.top_k = top_k or rag_config.TOP_K
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all components"""
        if self._initialized:
            return

        print("Initializing RAG pipeline...")
        self.vector_store.initialize()
        self.base_model.load()
        self._initialized = True
        print("RAG pipeline ready!")

    def retrieve(self, query: str, n_results: int = None) -> tuple:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            n_results: Number of documents to retrieve

        Returns:
            Tuple of (documents, metadatas, latency_ms)
        """
        if not self.vector_store.is_initialized():
            self.vector_store.initialize()

        n_results = n_results or self.top_k
        result = self.vector_store.query(query, n_results=n_results)

        return result.documents, result.metadatas, result.latency_ms

    def generate(
        self,
        question: str,
        table: Optional[str] = None,
        n_results: int = None,
        max_new_tokens: int = 512
    ) -> RAGResponse:
        """
        Full RAG pipeline: retrieve then generate.

        Args:
            question: Question to answer
            table: Optional table data to include
            n_results: Number of documents to retrieve
            max_new_tokens: Maximum tokens to generate

        Returns:
            RAGResponse with answer, documents, and metrics
        """
        if not self._initialized:
            self.initialize()

        total_start = time.perf_counter()

        # Step 1: Retrieve relevant documents
        docs, metas, retrieval_latency = self.retrieve(question, n_results)

        # Step 2: Generate answer with context
        gen_start = time.perf_counter()
        response = self.base_model.generate_with_context(
            question=question,
            context_docs=docs,
            table=table,
            max_new_tokens=max_new_tokens
        )
        gen_latency = (time.perf_counter() - gen_start) * 1000

        total_latency = (time.perf_counter() - total_start) * 1000

        return RAGResponse(
            answer=response.answer,
            retrieved_docs=docs,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=gen_latency,
            total_latency_ms=total_latency,
            num_docs_retrieved=len(docs),
            tokens_generated=response.tokens_generated
        )

    def generate_sentiment(
        self,
        text: str,
        n_results: int = 3
    ) -> RAGResponse:
        """
        RAG-based sentiment classification.

        Args:
            text: Text to classify
            n_results: Number of example documents to retrieve

        Returns:
            RAGResponse with sentiment classification
        """
        if not self._initialized:
            self.initialize()

        total_start = time.perf_counter()

        # Retrieve relevant sentiment examples
        query = f"financial sentiment classification example: {text}"
        docs, metas, retrieval_latency = self.retrieve(query, n_results)

        # Generate sentiment with context
        gen_start = time.perf_counter()
        response = self.base_model.generate_for_sentiment(
            text=text,
            context_docs=docs
        )
        gen_latency = (time.perf_counter() - gen_start) * 1000

        total_latency = (time.perf_counter() - total_start) * 1000

        return RAGResponse(
            answer=response.answer,
            retrieved_docs=docs,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=gen_latency,
            total_latency_ms=total_latency,
            num_docs_retrieved=len(docs),
            tokens_generated=response.tokens_generated
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None) -> int:
        """Add documents to the knowledge base"""
        return self.vector_store.add_documents(documents, metadatas)

    def get_document_count(self) -> int:
        """Get number of documents in knowledge base"""
        return self.vector_store.count()

    def is_initialized(self) -> bool:
        """Check if pipeline is initialized"""
        return self._initialized


# Singleton instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline instance"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
