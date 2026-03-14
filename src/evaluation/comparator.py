"""
Model Comparator for side-by-side evaluation
Compares Fine-Tuned, RAG, and Hybrid approaches in real-time
"""
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models.finqa_model import FinQAModel, get_finqa_model, FinQAResponse
from ..models.finbert_model import FinBERTModel, get_finbert_model, SentimentResult
from ..models.base_model import BaseModel, get_base_model
from ..models.hybrid_model import HybridModel, get_hybrid_model, HybridResponse
from ..rag.rag_pipeline import RAGPipeline, get_rag_pipeline, RAGResponse
from .metrics import (
    compute_numerical_accuracy,
    compute_sentiment_accuracy,
    compute_reasoning_quality,
    compute_token_efficiency,
    MetricResult
)


@dataclass
class ComparisonResult:
    """Result of comparing all three approaches"""
    # Responses
    finetuned_response: Any  # FinQAResponse or SentimentResult
    rag_response: RAGResponse
    hybrid_response: HybridResponse

    # Timing (ms)
    finetuned_latency: float
    rag_latency: float
    hybrid_latency: float

    # Metrics
    finetuned_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    rag_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    hybrid_metrics: Dict[str, MetricResult] = field(default_factory=dict)

    # Task info
    task_type: str = "numerical_reasoning"
    question: str = ""
    expected_answer: str = ""


class ModelComparator:
    """
    Compare Fine-Tuned, RAG, and Hybrid models side-by-side.
    Supports both numerical reasoning (FinQA) and sentiment analysis.
    """

    def __init__(
        self,
        finqa_model: FinQAModel = None,
        finbert_model: FinBERTModel = None,
        rag_pipeline: RAGPipeline = None,
        hybrid_model: HybridModel = None
    ):
        self.finqa_model = finqa_model
        self.finbert_model = finbert_model
        self.rag_pipeline = rag_pipeline
        self.hybrid_model = hybrid_model
        self._initialized = False

    def initialize(self, task_type: str = "numerical_reasoning") -> None:
        """
        Initialize models for comparison.

        Args:
            task_type: 'numerical_reasoning' or 'sentiment'
        """
        if self._initialized:
            return

        print(f"Initializing comparator for {task_type}...")

        if task_type == "numerical_reasoning":
            # Load FinQA and Hybrid models
            if self.finqa_model is None:
                self.finqa_model = get_finqa_model()
            self.finqa_model.load()

            if self.hybrid_model is None:
                self.hybrid_model = get_hybrid_model()
            self.hybrid_model.load()

        elif task_type == "sentiment":
            # Load FinBERT
            if self.finbert_model is None:
                self.finbert_model = get_finbert_model()
            self.finbert_model.load()

        # Initialize RAG pipeline
        if self.rag_pipeline is None:
            self.rag_pipeline = get_rag_pipeline()
        self.rag_pipeline.initialize()

        self._initialized = True
        print("Comparator ready!")

    def compare_numerical_reasoning(
        self,
        table: str,
        text: str,
        question: str,
        expected_answer: str = "",
        run_parallel: bool = False
    ) -> ComparisonResult:
        """
        Compare all three approaches on a numerical reasoning task.

        Args:
            table: Financial table data
            text: Context text
            question: Question to answer
            expected_answer: Ground truth answer for metrics
            run_parallel: Whether to run models in parallel

        Returns:
            ComparisonResult with all responses and metrics
        """
        if not self._initialized:
            self.initialize("numerical_reasoning")

        if run_parallel:
            return self._compare_parallel(table, text, question, expected_answer)
        else:
            return self._compare_sequential(table, text, question, expected_answer)

    def _compare_sequential(
        self,
        table: str,
        text: str,
        question: str,
        expected_answer: str
    ) -> ComparisonResult:
        """Run comparisons sequentially"""

        # 1. Fine-tuned model
        print("Running Fine-tuned model...")
        ft_start = time.perf_counter()
        ft_response = self.finqa_model.generate(table, text, question)
        ft_latency = (time.perf_counter() - ft_start) * 1000

        # 2. RAG pipeline
        print("Running RAG pipeline...")
        rag_start = time.perf_counter()
        rag_response = self.rag_pipeline.generate(question, table=table)
        rag_latency = (time.perf_counter() - rag_start) * 1000

        # 3. Hybrid model
        print("Running Hybrid model...")
        hybrid_start = time.perf_counter()
        hybrid_response = self.hybrid_model.generate(table, text, question)
        hybrid_latency = (time.perf_counter() - hybrid_start) * 1000

        # Compute metrics
        ft_metrics = self._compute_numerical_metrics(
            ft_response.answer, expected_answer, ft_response.reasoning_steps, ft_response.tokens_generated
        )
        rag_metrics = self._compute_numerical_metrics(
            rag_response.answer, expected_answer, rag_response.answer, rag_response.tokens_generated
        )
        hybrid_metrics = self._compute_numerical_metrics(
            hybrid_response.answer, expected_answer, hybrid_response.reasoning_steps, hybrid_response.tokens_generated
        )

        return ComparisonResult(
            finetuned_response=ft_response,
            rag_response=rag_response,
            hybrid_response=hybrid_response,
            finetuned_latency=ft_latency,
            rag_latency=rag_latency,
            hybrid_latency=hybrid_latency,
            finetuned_metrics=ft_metrics,
            rag_metrics=rag_metrics,
            hybrid_metrics=hybrid_metrics,
            task_type="numerical_reasoning",
            question=question,
            expected_answer=expected_answer
        )

    def _compare_parallel(
        self,
        table: str,
        text: str,
        question: str,
        expected_answer: str
    ) -> ComparisonResult:
        """Run comparisons in parallel using threads"""

        results = {}

        def run_finetuned():
            start = time.perf_counter()
            response = self.finqa_model.generate(table, text, question)
            latency = (time.perf_counter() - start) * 1000
            return ("finetuned", response, latency)

        def run_rag():
            start = time.perf_counter()
            response = self.rag_pipeline.generate(question, table=table)
            latency = (time.perf_counter() - start) * 1000
            return ("rag", response, latency)

        def run_hybrid():
            start = time.perf_counter()
            response = self.hybrid_model.generate(table, text, question)
            latency = (time.perf_counter() - start) * 1000
            return ("hybrid", response, latency)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_finetuned),
                executor.submit(run_rag),
                executor.submit(run_hybrid)
            ]

            for future in as_completed(futures):
                name, response, latency = future.result()
                results[name] = (response, latency)

        # Build result
        ft_response, ft_latency = results["finetuned"]
        rag_response, rag_latency = results["rag"]
        hybrid_response, hybrid_latency = results["hybrid"]

        # Compute metrics
        ft_metrics = self._compute_numerical_metrics(
            ft_response.answer, expected_answer, ft_response.reasoning_steps, ft_response.tokens_generated
        )
        rag_metrics = self._compute_numerical_metrics(
            rag_response.answer, expected_answer, rag_response.answer, rag_response.tokens_generated
        )
        hybrid_metrics = self._compute_numerical_metrics(
            hybrid_response.answer, expected_answer, hybrid_response.reasoning_steps, hybrid_response.tokens_generated
        )

        return ComparisonResult(
            finetuned_response=ft_response,
            rag_response=rag_response,
            hybrid_response=hybrid_response,
            finetuned_latency=ft_latency,
            rag_latency=rag_latency,
            hybrid_latency=hybrid_latency,
            finetuned_metrics=ft_metrics,
            rag_metrics=rag_metrics,
            hybrid_metrics=hybrid_metrics,
            task_type="numerical_reasoning",
            question=question,
            expected_answer=expected_answer
        )

    def _compute_numerical_metrics(
        self,
        answer: str,
        expected: str,
        reasoning: str,
        tokens: int
    ) -> Dict[str, MetricResult]:
        """Compute metrics for numerical reasoning task"""
        metrics = {}

        if expected:
            metrics["accuracy"] = compute_numerical_accuracy(answer, expected)

        metrics["reasoning_quality"] = compute_reasoning_quality(reasoning)
        metrics["token_efficiency"] = compute_token_efficiency(tokens, len(answer))

        return metrics

    def compare_sentiment(
        self,
        text: str,
        expected_label: str = ""
    ) -> Dict[str, Any]:
        """
        Compare FinBERT vs RAG for sentiment classification.

        Args:
            text: Financial text to classify
            expected_label: Ground truth sentiment

        Returns:
            Dict with both results and metrics
        """
        if not self.finbert_model or not self.finbert_model.is_loaded():
            if self.finbert_model is None:
                self.finbert_model = get_finbert_model()
            self.finbert_model.load()

        if not self.rag_pipeline or not self.rag_pipeline.is_initialized():
            if self.rag_pipeline is None:
                self.rag_pipeline = get_rag_pipeline()
            self.rag_pipeline.initialize()

        # Run FinBERT
        finbert_start = time.perf_counter()
        finbert_result = self.finbert_model.predict(text)
        finbert_latency = (time.perf_counter() - finbert_start) * 1000

        # Run RAG
        rag_start = time.perf_counter()
        rag_result = self.rag_pipeline.generate_sentiment(text)
        rag_latency = (time.perf_counter() - rag_start) * 1000

        # Compute metrics
        metrics = {}
        if expected_label:
            metrics["finbert_accuracy"] = compute_sentiment_accuracy(
                finbert_result.label, expected_label
            )
            metrics["rag_accuracy"] = compute_sentiment_accuracy(
                rag_result.answer, expected_label
            )

        return {
            "finbert": {
                "result": finbert_result,
                "latency_ms": finbert_latency
            },
            "rag": {
                "result": rag_result,
                "latency_ms": rag_latency
            },
            "metrics": metrics
        }

    def is_initialized(self) -> bool:
        """Check if comparator is initialized"""
        return self._initialized


# Singleton instance
_comparator: Optional[ModelComparator] = None


def get_comparator() -> ModelComparator:
    """Get or create the model comparator instance"""
    global _comparator
    if _comparator is None:
        _comparator = ModelComparator()
    return _comparator
