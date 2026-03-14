"""
Real RAG Engine using sentence-transformers + ChromaDB.
Loads actual financial documents, embeds them, and provides
similarity-based retrieval.
"""
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    documents: List[str]
    sources: List[str]
    distances: List[float]
    latency_ms: float
    num_chunks_searched: int


class RAGEngine:
    """Singleton RAG engine with real embeddings and vector search."""

    _instance: Optional["RAGEngine"] = None

    def __init__(self):
        self._embedder = None
        self._collection = None
        self._client = None
        self._initialized = False
        self._num_chunks = 0

    @classmethod
    def get_instance(cls) -> "RAGEngine":
        if cls._instance is None:
            cls._instance = RAGEngine()
        return cls._instance

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def num_chunks(self) -> int:
        return self._num_chunks

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self, docs_dir: Optional[str] = None):
        """Load documents, embed, and store in ChromaDB."""
        if self._initialized:
            return

        if docs_dir is None:
            # Try several paths (works both inside Docker and locally)
            candidates = [
                Path(__file__).parent.parent / "data" / "documents",
                Path("data/documents"),
                Path("/app/data/documents"),
            ]
            for p in candidates:
                if p.exists() and any(p.iterdir()):
                    docs_dir = str(p)
                    break
            else:
                raise FileNotFoundError(
                    "No documents directory found. "
                    "Expected data/documents/ with .txt files."
                )

        from sentence_transformers import SentenceTransformer
        import chromadb

        print("[RAG] Loading embedding model...")
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        print("[RAG] Initializing ChromaDB (in-memory)...")
        self._client = chromadb.Client()

        # Delete existing collection if any (fresh start)
        try:
            self._client.delete_collection("financial_docs")
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name="financial_docs",
            metadata={"hnsw:space": "cosine"},
        )

        # Load and chunk documents
        print(f"[RAG] Loading documents from {docs_dir}...")
        raw_docs = self._load_documents(docs_dir)
        chunks = self._chunk_documents(raw_docs, chunk_size=300, overlap=50)
        print(f"[RAG] Created {len(chunks)} chunks from {len(raw_docs)} documents")

        if not chunks:
            raise ValueError("No document chunks created. Check data/documents/")

        # Embed all chunks
        print("[RAG] Embedding chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self._embedder.encode(texts, show_progress_bar=False)

        # Store in ChromaDB
        self._collection.add(
            documents=texts,
            metadatas=[{"source": c["source"], "chunk_idx": c["chunk_idx"]}
                       for c in chunks],
            embeddings=embeddings.tolist(),
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )

        self._num_chunks = len(chunks)
        self._initialized = True
        print(f"[RAG] Ready. {self._num_chunks} chunks indexed.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Retrieve the most relevant document chunks for a query."""
        if not self._initialized:
            self.initialize()

        start = time.perf_counter()

        query_embedding = self._embedder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        sources = [m.get("source", "unknown") for m in metadatas]

        return RetrievalResult(
            documents=documents,
            sources=sources,
            distances=distances,
            latency_ms=round(elapsed_ms, 1),
            num_chunks_searched=self._num_chunks,
        )

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------
    def _load_documents(self, docs_dir: str) -> List[Dict]:
        """Load all .txt files from the documents directory."""
        docs_path = Path(docs_dir)
        documents = []

        for fpath in sorted(docs_path.glob("*.txt")):
            text = fpath.read_text(encoding="utf-8").strip()
            if text:
                documents.append({
                    "text": text,
                    "source": fpath.name,
                })

        return documents

    def _chunk_documents(self, documents: List[Dict],
                         chunk_size: int = 300,
                         overlap: int = 50) -> List[Dict]:
        """Split documents into overlapping chunks by word count."""
        chunks = []
        for doc in documents:
            words = doc["text"].split()
            if len(words) <= chunk_size:
                chunks.append({
                    "text": doc["text"],
                    "source": doc["source"],
                    "chunk_idx": 0,
                })
            else:
                idx = 0
                start = 0
                while start < len(words):
                    end = min(start + chunk_size, len(words))
                    chunk_text = " ".join(words[start:end])
                    chunks.append({
                        "text": chunk_text,
                        "source": doc["source"],
                        "chunk_idx": idx,
                    })
                    idx += 1
                    start += chunk_size - overlap
        return chunks
