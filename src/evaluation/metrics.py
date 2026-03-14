"""
Evaluation metrics for comparing Fine-Tuned vs RAG vs Hybrid
"""
import time
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result of metric computation"""
    name: str
    value: float
    unit: str
    description: str


def compute_latency(start_time: float, end_time: float) -> MetricResult:
    """Compute latency in milliseconds"""
    latency_ms = (end_time - start_time) * 1000
    return MetricResult(
        name="latency",
        value=latency_ms,
        unit="ms",
        description="Response time"
    )


def extract_number(text: str) -> Optional[float]:
    """Extract the first number from text"""
    # Match numbers with optional decimals and percentage signs
    patterns = [
        r'[-+]?\d*\.?\d+%',  # Percentage
        r'[-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?',  # Currency/number with commas
        r'[-+]?\d*\.?\d+',  # Plain number
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            num_str = match.group().replace('%', '').replace('$', '').replace(',', '')
            try:
                return float(num_str)
            except ValueError:
                continue
    return None


def compute_numerical_accuracy(predicted: str, expected: str, tolerance: float = 0.01) -> MetricResult:
    """
    Compute accuracy for numerical answers.

    Args:
        predicted: Model's predicted answer
        expected: Ground truth answer
        tolerance: Acceptable relative error (default 1%)

    Returns:
        MetricResult with accuracy (0 or 1)
    """
    pred_num = extract_number(predicted)
    exp_num = extract_number(expected)

    if pred_num is None or exp_num is None:
        # Fall back to string matching
        is_correct = predicted.strip().lower() == expected.strip().lower()
        accuracy = 1.0 if is_correct else 0.0
    else:
        # Numerical comparison with tolerance
        if exp_num == 0:
            is_correct = abs(pred_num) < tolerance
        else:
            relative_error = abs(pred_num - exp_num) / abs(exp_num)
            is_correct = relative_error <= tolerance

        accuracy = 1.0 if is_correct else 0.0

    return MetricResult(
        name="numerical_accuracy",
        value=accuracy,
        unit="",
        description="Correct numerical answer"
    )


def compute_sentiment_accuracy(predicted: str, expected: str) -> MetricResult:
    """
    Compute accuracy for sentiment classification.

    Args:
        predicted: Model's predicted sentiment
        expected: Ground truth sentiment

    Returns:
        MetricResult with accuracy (0 or 1)
    """
    # Normalize labels
    pred_lower = predicted.lower()
    exp_lower = expected.lower()

    # Check for sentiment keywords
    if "positive" in pred_lower and exp_lower == "positive":
        accuracy = 1.0
    elif "negative" in pred_lower and exp_lower == "negative":
        accuracy = 1.0
    elif "neutral" in pred_lower and exp_lower == "neutral":
        accuracy = 1.0
    else:
        accuracy = 0.0

    return MetricResult(
        name="sentiment_accuracy",
        value=accuracy,
        unit="",
        description="Correct sentiment classification"
    )


def compute_token_efficiency(tokens_generated: int, answer_length: int) -> MetricResult:
    """
    Compute token efficiency (useful answer per token).

    Args:
        tokens_generated: Total tokens generated
        answer_length: Length of actual answer

    Returns:
        MetricResult with efficiency ratio
    """
    if tokens_generated == 0:
        efficiency = 0.0
    else:
        efficiency = answer_length / tokens_generated

    return MetricResult(
        name="token_efficiency",
        value=efficiency,
        unit="",
        description="Answer length / tokens generated"
    )


def compute_reasoning_quality(response: str) -> MetricResult:
    """
    Heuristic assessment of reasoning quality.
    Checks for step-by-step reasoning indicators.

    Args:
        response: Full model response

    Returns:
        MetricResult with quality score (0-1)
    """
    score = 0.0
    response_lower = response.lower()

    # Check for step indicators
    step_patterns = [
        r'step\s*\d',
        r'\d\)',
        r'\d\.',
        r'first[,:]',
        r'second[,:]',
        r'then[,:]',
        r'finally[,:]',
        r'therefore',
        r'thus',
        r'hence'
    ]

    steps_found = sum(1 for p in step_patterns if re.search(p, response_lower))
    score += min(steps_found * 0.15, 0.45)  # Up to 0.45 for step structure

    # Check for calculations
    calc_patterns = [r'=', r'\+', r'-', r'\*', r'/', r'×', r'÷']
    calcs_found = sum(1 for p in calc_patterns if p in response)
    score += min(calcs_found * 0.1, 0.3)  # Up to 0.3 for calculations

    # Check for clear answer statement
    if any(x in response_lower for x in ['answer:', 'final answer:', 'result:']):
        score += 0.25

    return MetricResult(
        name="reasoning_quality",
        value=min(score, 1.0),
        unit="",
        description="Quality of reasoning steps"
    )


def aggregate_metrics(metrics_list: List[Dict[str, MetricResult]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple examples.

    Args:
        metrics_list: List of metric dicts from multiple examples

    Returns:
        Dict with averaged metrics
    """
    if not metrics_list:
        return {}

    aggregated = {}

    # Get all metric names
    all_names = set()
    for m in metrics_list:
        all_names.update(m.keys())

    # Compute averages
    for name in all_names:
        values = [m[name].value for m in metrics_list if name in m]
        if values:
            aggregated[name] = sum(values) / len(values)

    return aggregated
