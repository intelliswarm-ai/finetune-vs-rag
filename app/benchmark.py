"""
Benchmark Runner -- Two controlled experiments on two architectures.

Section 1: BERT 110M (Sentiment Classification)
  - Base BERT vs FinBERT (fine-tuned) vs BERT + RAG vs Hybrid (FinBERT + RAG)

Section 2: Llama2 7B (Numerical Reasoning)
  - Base Llama2-7B vs FinQA-7B (fine-tuned) vs Llama2-7B + RAG vs FinQA-7B + RAG (hybrid)

Usage:
    python app/benchmark.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_PATH = Path(__file__).parent.parent / "data" / "benchmark_results.json"
TEST_CASES_PATH = Path(__file__).parent.parent / "data" / "benchmark_test_cases.json"


# -------------------------------------------------------------------------
# Section 1: BERT 110M Sentiment
# -------------------------------------------------------------------------
def run_bert_sentiment_benchmark():
    """Base BERT vs FinBERT vs BERT+RAG vs Hybrid -- all 110M params."""
    from demo_utils import (
        run_finbert, run_base_bert_sentiment, run_rag_sentiment,
        run_hybrid_sentiment, finbert_available, bert_base_available,
    )

    with open(TEST_CASES_PATH) as f:
        cases = json.load(f)["sentiment"]

    model_names = ["base", "finbert", "rag", "hybrid"]
    print(f"  Running {len(cases)} sentiment cases x {len(model_names)} models (BERT 110M)...")
    results = []

    for i, case in enumerate(cases):
        text, expected = case["text"], case["label"]
        row = {"id": case["id"], "text": text, "expected": expected,
               "category": case["category"]}

        for name, fn in [("base", run_base_bert_sentiment),
                         ("finbert", run_finbert),
                         ("rag", run_rag_sentiment),
                         ("hybrid", run_hybrid_sentiment)]:
            try:
                r = fn(text)
                row[f"{name}_label"] = r.label
                row[f"{name}_confidence"] = r.confidence
                row[f"{name}_latency_ms"] = r.latency_ms
                row[f"{name}_correct"] = r.label == expected
            except Exception:
                row[f"{name}_label"] = "error"
                row[f"{name}_correct"] = False

        results.append(row)

        marks = "  ".join(
            f"{n}={row.get(f'{n}_label','?'):>8}[{'Y' if row.get(f'{n}_correct') else 'N'}]"
            for n in model_names
        )
        print(f"    [{i+1:>2}/{len(cases)}] exp={expected:>8}  {marks}")

    return results


# -------------------------------------------------------------------------
# Section 2: Llama2 7B Numerical Reasoning
# -------------------------------------------------------------------------
NUMERICAL_CASES = [
    {
        "id": "n01",
        "table": "| Segment | 2023 | 2022 |\n|---|---|---|\n| Consumer | $12,450 | $11,200 |\n| Commercial | $8,320 | $7,890 |\n| Investment | $5,180 | $6,240 |",
        "context": "Investment Banking revenue decreased due to lower trading volumes.",
        "question": "What was the percentage change in total revenue from 2022 to 2023?",
        "expected": "2.45",
        "category": "multi_step",
    },
    {
        "id": "n02",
        "table": "| Item | 2023 | 2022 |\n|---|---|---|\n| Total Debt | $52,300 | $48,900 |\n| Equity | $47,400 | $43,800 |",
        "context": "The company maintained a strong capital position.",
        "question": "Calculate the debt-to-equity ratio for 2023.",
        "expected": "1.10",
        "category": "single_step",
    },
    {
        "id": "n03",
        "table": "| Metric | Q4 2023 | Q4 2022 |\n|---|---|---|\n| Revenue | $19,800 | $18,600 |\n| OpEx | $11,200 | $10,400 |",
        "context": "Net interest income benefited from higher rates.",
        "question": "What is the efficiency ratio (OpEx/Revenue) for Q4 2023?",
        "expected": "56.57",
        "category": "single_step",
    },
    {
        "id": "n04",
        "table": "| Item | 2023 | 2022 |\n|---|---|---|\n| Net Income | $8,200 | $7,500 |\n| Avg Equity | $45,600 | $42,100 |",
        "context": "Profitability improved year over year.",
        "question": "Calculate the Return on Equity (ROE = Net Income / Avg Equity) for both years.",
        "expected": "17.98",
        "category": "multi_step",
    },
    {
        "id": "n05",
        "table": "| Region | 2023 | 2022 |\n|---|---|---|\n| North America | $18,200 | $16,950 |\n| Europe | $8,400 | $9,130 |\n| Asia Pacific | $6,900 | $5,990 |",
        "context": "Asia Pacific saw strong growth while Europe contracted.",
        "question": "What was the year-over-year growth rate for Asia Pacific?",
        "expected": "15.19",
        "category": "single_step",
    },
]


def _extract_number(text: str):
    """Try to extract a key number from model output."""
    import re
    # Look for percentages first
    pcts = re.findall(r'(\d+\.?\d*)\s*%', text)
    if pcts:
        return pcts[0]
    # Look for ratios (e.g., 1.10)
    ratios = re.findall(r'(\d+\.\d{2,})', text)
    if ratios:
        return ratios[0]
    # Any number
    nums = re.findall(r'(\d+\.?\d+)', text)
    return nums[0] if nums else None


def _check_numerical(model_answer: str, expected: str, tolerance: float = 0.05):
    """Check if model's answer contains the expected number within tolerance."""
    extracted = _extract_number(model_answer)
    if extracted is None:
        return False, None
    try:
        got = float(extracted)
        exp = float(expected)
        if exp == 0:
            return abs(got) < tolerance, extracted
        return abs(got - exp) / abs(exp) <= tolerance, extracted
    except ValueError:
        return False, extracted


def run_llama_numerical_benchmark():
    """Llama2-7B: base vs expert prompt vs RAG vs hybrid on numerical reasoning."""
    from demo_utils import (
        call_finetuned_model, call_base_model, call_rag_model,
        call_hybrid_model, has_llm, LLM_MODEL,
    )

    if not has_llm():
        print("  [SKIP] Ollama not available -- skipping Llama2 benchmark")
        return []

    model_names = ["base", "finetuned", "rag", "hybrid"]
    model_fns = {
        "base": call_base_model,
        "finetuned": call_finetuned_model,
        "rag": call_rag_model,
        "hybrid": call_hybrid_model,
    }

    print(f"  Running {len(NUMERICAL_CASES)} numerical cases x {len(model_names)} approaches ({LLM_MODEL})...")
    results = []

    for i, case in enumerate(NUMERICAL_CASES):
        q, table, ctx = case["question"], case["table"], case["context"]
        expected = case["expected"]

        row = {"id": case["id"], "question": q, "expected": expected,
               "category": case["category"]}

        for name in model_names:
            try:
                r = model_fns[name](q, table, ctx)
                correct, extracted = _check_numerical(r.answer, expected)
                row[f"{name}_answer"] = r.answer[:200]
                row[f"{name}_extracted"] = extracted
                row[f"{name}_correct"] = correct
                row[f"{name}_latency_ms"] = r.latency_ms
            except Exception as e:
                row[f"{name}_correct"] = False
                row[f"{name}_answer"] = str(e)[:100]

        results.append(row)

        marks = "  ".join(
            f"{n}={'Y' if row.get(f'{n}_correct') else 'N'}"
            for n in model_names
        )
        print(f"    [{i+1:>2}/{len(NUMERICAL_CASES)}] expected={expected:>6}  {marks}")

    return results


# -------------------------------------------------------------------------
# Streaming (case-by-case) versions for live UI
# -------------------------------------------------------------------------
SENTIMENT_MODEL_NAMES = ["base", "finbert", "rag", "hybrid"]
NUMERICAL_MODEL_NAMES = ["base", "finetuned", "rag", "hybrid"]


def get_sentiment_cases():
    """Return the sentiment test cases list."""
    with open(TEST_CASES_PATH) as f:
        return json.load(f)["sentiment"]


def get_numerical_cases():
    """Return the numerical reasoning test cases list."""
    return NUMERICAL_CASES


def run_single_sentiment_case(case):
    """Run all 4 models on a single sentiment case. Returns result dict."""
    from demo_utils import (
        run_finbert, run_base_bert_sentiment, run_rag_sentiment,
        run_hybrid_sentiment,
    )

    text, expected = case["text"], case["label"]
    row = {"id": case["id"], "text": text, "expected": expected,
           "category": case["category"]}

    for name, fn in [("base", run_base_bert_sentiment),
                     ("finbert", run_finbert),
                     ("rag", run_rag_sentiment),
                     ("hybrid", run_hybrid_sentiment)]:
        try:
            r = fn(text)
            row[f"{name}_label"] = r.label
            row[f"{name}_confidence"] = r.confidence
            row[f"{name}_latency_ms"] = r.latency_ms
            row[f"{name}_correct"] = r.label == expected
        except Exception:
            row[f"{name}_label"] = "error"
            row[f"{name}_correct"] = False

    return row


def run_single_numerical_case(case):
    """Run all 4 models on a single numerical case. Returns result dict."""
    row = init_numerical_row(case)
    for name in NUMERICAL_MODEL_NAMES:
        run_single_numerical_model(row, case, name)
    return row


def init_numerical_row(case):
    """Initialize a result row for a numerical case (no model results yet)."""
    return {"id": case["id"], "question": case["question"],
            "expected": case["expected"], "category": case["category"]}


NUMERICAL_MODEL_FNS = {
    "base": "call_base_model",
    "finetuned": "call_finetuned_model",
    "rag": "call_rag_model",
    "hybrid": "call_hybrid_model",
}


def run_single_numerical_model(row, case, model_name):
    """Run ONE model on a numerical case, updating row in-place.

    Returns the model result dict with answer, correct, latency."""
    import demo_utils
    fn = getattr(demo_utils, NUMERICAL_MODEL_FNS[model_name])
    q, table, ctx = case["question"], case["table"], case["context"]
    expected = case["expected"]
    try:
        r = fn(q, table, ctx)
        correct, extracted = _check_numerical(r.answer, expected)
        row[f"{model_name}_answer"] = r.answer[:200]
        row[f"{model_name}_extracted"] = extracted
        row[f"{model_name}_correct"] = correct
        row[f"{model_name}_latency_ms"] = r.latency_ms
        return {"answer": r.answer[:200], "correct": correct,
                "extracted": extracted, "latency_ms": r.latency_ms}
    except Exception as e:
        row[f"{model_name}_correct"] = False
        row[f"{model_name}_answer"] = str(e)[:100]
        return {"answer": str(e)[:100], "correct": False,
                "extracted": None, "latency_ms": 0}


def compute_live_stats(results, model_keys):
    """Compute running accuracy and avg latency from partial results."""
    total = len(results)
    if total == 0:
        return {m: {"accuracy": 0, "correct": 0, "total": 0,
                     "avg_latency_ms": 0, "avg_confidence": 0}
                for m in model_keys}
    stats = {}
    for m in model_keys:
        correct = sum(1 for r in results if r.get(f"{m}_correct"))
        latencies = [r[f"{m}_latency_ms"] for r in results
                     if f"{m}_latency_ms" in r]
        confidences = [r[f"{m}_confidence"] for r in results
                       if f"{m}_confidence" in r]
        stats[m] = {
            "accuracy": round(correct / total * 100, 1),
            "correct": correct,
            "total": total,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0,
        }
    return stats


def compute_section_summary(results, model_keys):
    """Compute stats for a benchmark section."""
    total = len(results)
    if total == 0:
        return {}

    def _stats(prefix):
        correct = sum(1 for r in results if r.get(f"{prefix}_correct"))
        latencies = [r[f"{prefix}_latency_ms"] for r in results
                     if f"{prefix}_latency_ms" in r]
        confidences = [r[f"{prefix}_confidence"] for r in results
                       if f"{prefix}_confidence" in r]
        return {
            "accuracy": round(correct / total * 100, 1),
            "correct": correct,
            "total": total,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        }

    summary = {m: _stats(m) for m in model_keys}

    # Per-category
    categories = sorted(set(r.get("category", "other") for r in results))
    for cat in categories:
        cat_results = [r for r in results if r.get("category") == cat]
        n = len(cat_results)
        entry = {"total": n}
        for m in model_keys:
            c = sum(1 for r in cat_results if r.get(f"{m}_correct"))
            entry[f"{m}_accuracy"] = round(c / n * 100, 1)
        summary[f"category_{cat}"] = entry

    return summary


def run_full_benchmark():
    """Run both benchmark sections and save."""
    print("=" * 70)
    print("BENCHMARK: Two controlled experiments")
    print("=" * 70)

    # Section 1: BERT 110M
    print("\n--- Section 1: BERT 110M (Sentiment) ---")
    sent_results = run_bert_sentiment_benchmark()
    sent_summary = compute_section_summary(
        sent_results, ["base", "finbert", "rag", "hybrid"])

    # Section 2: Llama2 7B
    print("\n--- Section 2: Llama2 7B (Numerical Reasoning) ---")
    num_results = run_llama_numerical_benchmark()
    num_summary = compute_section_summary(
        num_results, ["base", "finetuned", "rag", "hybrid"])

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sections": {
            "bert_110m_sentiment": {
                "title": "BERT 110M -- Sentiment Classification",
                "architecture": "BERT-base-uncased (110M parameters)",
                "models": ["base", "finbert", "rag", "hybrid"],
                "model_labels": {
                    "base": "Base BERT (no FT, no RAG)",
                    "finbert": "FinBERT (fine-tuned)",
                    "rag": "BERT + RAG (retrieval + voting)",
                    "hybrid": "FinBERT + RAG (hybrid)",
                },
                "summary": sent_summary,
                "results": sent_results,
            },
            "llama2_7b_numerical": {
                "title": "Llama2 7B -- Numerical Reasoning",
                "architecture": "Llama2-7B (7B parameters)",
                "models": ["base", "finetuned", "rag", "hybrid"],
                "model_labels": {
                    "base": "Base Llama2-7B",
                    "finetuned": "FinQA-7B (fine-tuned Llama2-7B)",
                    "rag": "Llama2-7B + RAG (base model)",
                    "hybrid": "FinQA-7B + RAG (fine-tuned + retrieval)",
                },
                "summary": num_summary,
                "results": num_results,
            },
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Print summaries
    for section_key, section in output["sections"].items():
        s = section["summary"]
        labels = section["model_labels"]
        print(f"\n{'='*60}")
        print(f"{section['title']} ({section['architecture']})")
        print(f"{'='*60}")
        for m in section["models"]:
            ms = s.get(m, {})
            print(f"  {labels.get(m,m):>40}: {ms.get('accuracy',0):>5}% "
                  f"({ms.get('correct',0)}/{ms.get('total',0)})")

    print(f"\nResults saved to {RESULTS_PATH}")
    return output


if __name__ == "__main__":
    run_full_benchmark()
