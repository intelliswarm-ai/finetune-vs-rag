"""
Configuration for Fine-Tuning vs RAG Demo
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for model loading"""
    # Fine-tuned models
    FINQA_MODEL_ID: str = "truocpham/FinQA-7B-Instruct-v0.1"
    FINBERT_MODEL_ID: str = "ProsusAI/finbert"
    FINBERT_TONE_MODEL_ID: str = "yiyanghkust/finbert-tone"

    # Base model for RAG
    BASE_MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # Embedding model
    EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Quantization settings
    USE_4BIT: bool = True
    USE_8BIT: bool = False

    # Generation settings
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.95
    DO_SAMPLE: bool = True


@dataclass
class DatasetConfig:
    """Configuration for datasets"""
    # FinQA - Primary benchmark
    FINQA_DATASET_ID: str = "ibm-research/finqa"

    # Financial PhraseBank - Sentiment
    PHRASEBANK_DATASET_ID: str = "takala/financial_phrasebank"

    # SEC Filings - For RAG knowledge base
    SEC_DATASET_ID: str = "PleIAs/SEC"

    # Number of samples for demo
    NUM_DEMO_SAMPLES: int = 20
    NUM_RAG_DOCUMENTS: int = 100


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    # Vector store
    COLLECTION_NAME: str = "financial_docs"
    PERSIST_DIRECTORY: str = str(DATA_DIR / "chroma_db")

    # Retrieval settings
    TOP_K: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Similarity threshold
    SIMILARITY_THRESHOLD: float = 0.5


@dataclass
class AppConfig:
    """Configuration for Streamlit app"""
    PAGE_TITLE: str = "Fine-Tuning vs RAG: Financial Demo"
    PAGE_ICON: str = "📊"
    LAYOUT: str = "wide"

    # Demo settings
    SHOW_REASONING_STEPS: bool = True
    SHOW_LATENCY_METRICS: bool = True
    SHOW_CONFIDENCE_SCORES: bool = True


# Global config instances
model_config = ModelConfig()
dataset_config = DatasetConfig()
rag_config = RAGConfig()
app_config = AppConfig()


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment"""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def is_gpu_available() -> bool:
    """Check if CUDA GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """Get the appropriate device for model inference"""
    if is_gpu_available():
        return "cuda"
    return "cpu"
