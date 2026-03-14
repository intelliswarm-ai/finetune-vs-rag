"""
FinBERT Model for financial sentiment analysis
Model: ProsusAI/finbert
"""
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..config import model_config, get_device


@dataclass
class SentimentResult:
    """Result from sentiment analysis"""
    label: str  # positive, negative, neutral
    confidence: float
    all_scores: Dict[str, float]
    latency_ms: float
    model_name: str = "FinBERT"


class FinBERTModel:
    """
    Fine-tuned BERT model for financial sentiment analysis.
    Classifies text as positive, negative, or neutral.
    """

    LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or model_config.FINBERT_MODEL_ID
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer"""
        if self._loaded:
            return

        print(f"Loading FinBERT model: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        print("FinBERT model loaded successfully!")

    def predict(self, text: str) -> SentimentResult:
        """
        Predict sentiment for a single text.

        Args:
            text: Financial text to analyze

        Returns:
            SentimentResult with label, confidence, and metrics
        """
        if not self._loaded:
            self.load()

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict with timing
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Get predictions
        probs = probs.cpu().numpy()[0]
        predicted_class = int(probs.argmax())
        confidence = float(probs[predicted_class])

        # Build scores dict
        all_scores = {
            self.LABEL_MAP[i]: float(probs[i])
            for i in range(len(probs))
        }

        return SentimentResult(
            label=self.LABEL_MAP[predicted_class],
            confidence=confidence,
            all_scores=all_scores,
            latency_ms=latency_ms
        )

    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Predict sentiment for multiple texts.

        Args:
            texts: List of financial texts to analyze

        Returns:
            List of SentimentResult objects
        """
        if not self._loaded:
            self.load()

        results = []

        # Tokenize all texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict with timing
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000
        per_item_latency = total_latency_ms / len(texts)

        # Process each result
        probs = probs.cpu().numpy()

        for i, prob in enumerate(probs):
            predicted_class = int(prob.argmax())
            confidence = float(prob[predicted_class])

            all_scores = {
                self.LABEL_MAP[j]: float(prob[j])
                for j in range(len(prob))
            }

            results.append(SentimentResult(
                label=self.LABEL_MAP[predicted_class],
                confidence=confidence,
                all_scores=all_scores,
                latency_ms=per_item_latency
            ))

        return results

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded

    def unload(self) -> None:
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Singleton instance
_finbert_model: Optional[FinBERTModel] = None


def get_finbert_model() -> FinBERTModel:
    """Get or create the FinBERT model instance"""
    global _finbert_model
    if _finbert_model is None:
        _finbert_model = FinBERTModel()
    return _finbert_model
