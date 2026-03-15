"""
Benchmark Runner -- Three controlled experiments on two architectures.

Section 1: BERT 110M (Sentiment Classification)
  - Base BERT vs FinBERT (fine-tuned) vs BERT + RAG vs Hybrid (FinBERT + RAG)

Section 2: Llama2 7B (Numerical Reasoning)
  - Base Llama2-7B vs FinQA-7B (fine-tuned) vs Llama2-7B + RAG vs FinQA-7B + RAG (hybrid)

Section 3: Llama2 7B (Financial Ratios)
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


def _is_year(s: str) -> bool:
    """Return True if string looks like a 4-digit year (1900-2099)."""
    try:
        n = float(s)
        return n == int(n) and 1900 <= int(n) <= 2099
    except (ValueError, OverflowError):
        return False


def _extract_number(text: str):
    """Try to extract a key number from model output, skipping years."""
    import re
    # Look for percentages first
    pcts = [p for p in re.findall(r'(\d+\.?\d*)\s*%', text) if not _is_year(p)]
    if pcts:
        return pcts[0]
    # Look for ratios (e.g., 1.10)
    ratios = [r for r in re.findall(r'(\d+\.\d{2,})', text) if not _is_year(r)]
    if ratios:
        return ratios[0]
    # Any number (skip years and dollar amounts in the thousands)
    nums = [n for n in re.findall(r'(\d+\.?\d+)', text) if not _is_year(n)]
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
                row[f"{name}_answer"] = r.answer[:500]
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
# Section 3: Llama2 7B Financial Ratios
# -------------------------------------------------------------------------
FINANCIAL_RATIO_CASES = [
    {
        "id": "fr01",
        "table": "| Item | 2023 |\n|---|---|\n| Net Income | $8,200M |\n| Revenue | $56,300M |\n| Total Assets (avg) | $187,600M |\n| Shareholders' Equity (avg) | $45,600M |",
        "context": "Analysts use the DuPont framework to decompose return on equity into profitability, efficiency, and leverage components.",
        "question": "Decompose the Return on Equity using the 3-component DuPont formula: ROE = (Net Income/Revenue) x (Revenue/Total Assets) x (Total Assets/Equity). What is the ROE percentage?",
        "expected": "17.98",
        "category": "profitability",
    },
    {
        "id": "fr02",
        "table": "| Item | 2023 | 2020 |\n|---|---|---|\n| Revenue | $56,300M | $42,100M |",
        "context": "The company achieved consistent revenue growth over the three-year period from 2020 to 2023.",
        "question": "Calculate the 3-year Compound Annual Growth Rate (CAGR) of revenue from 2020 to 2023. CAGR = (End/Start)^(1/n) - 1.",
        "expected": "10.17",
        "category": "efficiency",
    },
    {
        "id": "fr03",
        "table": "| Item | Q4 2023 | Q3 2023 |\n|---|---|---|\n| Current Assets | $125,400M | $118,200M |\n| Current Liabilities | $89,500M | $92,100M |",
        "context": "The bank improved its short-term liquidity position in Q4 through asset growth and liability reduction.",
        "question": "Calculate the working capital for both quarters, then compute the percentage change in working capital from Q3 to Q4.",
        "expected": "37.55",
        "category": "liquidity",
    },
    {
        "id": "fr04",
        "table": "| Item | 2023 |\n|---|---|\n| Short-term Debt | $12,400M |\n| Long-term Debt | $55,400M |\n| Common Stock | $18,200M |\n| Additional Paid-in Capital | $9,800M |\n| Retained Earnings | $19,400M |",
        "context": "The company's capital structure includes multiple debt and equity components.",
        "question": "Calculate the debt-to-equity ratio by first summing all debt components and all equity components, then dividing total debt by total equity.",
        "expected": "1.43",
        "category": "leverage",
    },
    {
        "id": "fr05",
        "table": "| Item | 2023 |\n|---|---|\n| Net Income | $8,200M |\n| Depreciation & Amortization | $3,400M |\n| Increase in Accounts Receivable | $1,200M |\n| Increase in Accounts Payable | $800M |\n| Current Liabilities | $89,500M |",
        "context": "Operating cash flow adjusts net income for non-cash items and working capital changes.",
        "question": "Calculate the operating cash flow ratio. First compute Operating Cash Flow = Net Income + D&A - Increase in AR + Increase in AP. Then divide by Current Liabilities.",
        "expected": "12.51",
        "category": "liquidity",
    },
    {
        "id": "fr06",
        "table": "| Item | 2023 |\n|---|---|\n| Revenue | $56,300M |\n| Cost of Goods Sold | $33,780M |\n| Operating Expenses | $8,450M |\n| Interest Expense | $3,800M |\n| Tax Expense | $2,568M |",
        "context": "Margin analysis reveals how much profit is retained at each stage of the income statement.",
        "question": "Calculate the gross margin, operating margin, and net profit margin. Then compute the spread between gross margin and net margin (gross margin minus net margin) in percentage points.",
        "expected": "26.32",
        "category": "profitability",
    },
    {
        "id": "fr07",
        "table": "| Item | 2023 |\n|---|---|\n| Net Income | $8,200M |\n| Shareholders' Equity (avg) | $45,600M |\n| Dividends Paid | $2,460M |",
        "context": "The sustainable growth rate estimates how fast a company can grow using only retained earnings without external financing.",
        "question": "Calculate the sustainable growth rate: SGR = ROE x (1 - Payout Ratio). First compute ROE (Net Income / Equity) and the payout ratio (Dividends / Net Income), then combine them.",
        "expected": "12.59",
        "category": "shareholder",
    },
    {
        "id": "fr08",
        "table": "| Item | 2023 | 2022 |\n|---|---|---|\n| EBIT | $15,200M | $13,800M |\n| Interest Expense | $3,800M | $4,200M |\n| Total Debt | $67,800M | $72,300M |\n| Shareholders' Equity | $47,400M | $43,800M |",
        "context": "Leverage analysis requires examining both coverage ratios and structural leverage together.",
        "question": "Calculate the interest coverage ratio (EBIT/Interest) and the debt-to-equity ratio (Debt/Equity) for both years. What is the percentage improvement in the interest coverage ratio from 2022 to 2023?",
        "expected": "21.74",
        "category": "leverage",
    },
]


FINANCIAL_RATIO_LABELS = {
    "base": "Base Llama2-7B",
    "finetuned": "FinQA-7B (fine-tuned Llama2-7B)",
    "rag": "Llama2-7B + RAG (base model)",
    "hybrid": "FinQA-7B + RAG (fine-tuned + retrieval)",
}


def _format_time(seconds):
    """Format seconds as 'Xm Ys'."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def _save_financial_ratio_results(results, model_names):
    """Save partial/complete financial ratio results to disk, preserving other sections."""
    existing = {}
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing_sections = existing.get("sections", {})
    summary = compute_section_summary(results, model_names) if results else {}

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sections": {
            "bert_110m_sentiment": existing_sections.get("bert_110m_sentiment", {
                "title": "BERT 110M -- Sentiment Classification",
                "architecture": "BERT-base-uncased (110M parameters)",
                "models": ["base", "finbert", "rag", "hybrid"],
                "model_labels": {"base": "Base BERT (no FT, no RAG)",
                                 "finbert": "FinBERT (fine-tuned)",
                                 "rag": "BERT + RAG (retrieval + voting)",
                                 "hybrid": "FinBERT + RAG (hybrid)"},
                "summary": {}, "results": [],
            }),
            "llama2_7b_numerical": existing_sections.get("llama2_7b_numerical", {
                "title": "Llama2 7B -- Numerical Reasoning",
                "architecture": "Llama2-7B (7B parameters)",
                "models": ["base", "finetuned", "rag", "hybrid"],
                "model_labels": {"base": "Base Llama2-7B",
                                 "finetuned": "FinQA-7B (fine-tuned Llama2-7B)",
                                 "rag": "Llama2-7B + RAG (base model)",
                                 "hybrid": "FinQA-7B + RAG (fine-tuned + retrieval)"},
                "summary": {}, "results": [],
            }),
            "llama2_7b_financial_ratios": {
                "title": "Llama2 7B -- Financial Ratios",
                "architecture": "Llama2-7B (7B parameters)",
                "models": list(model_names),
                "model_labels": FINANCIAL_RATIO_LABELS,
                "summary": summary,
                "results": results,
            },
        },
    }

    # Atomic write: write to temp file then rename
    tmp_path = RESULTS_PATH.with_suffix(".tmp")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2)
    tmp_path.replace(RESULTS_PATH)


def run_financial_ratio_benchmark():
    """Llama2-7B: base vs fine-tuned vs RAG vs hybrid on financial ratio calculations.

    Saves results incrementally after each case completes."""
    from demo_utils import (
        call_finetuned_model, call_base_model, call_rag_model,
        call_hybrid_model, has_llm, LLM_MODEL,
    )

    if not has_llm():
        print("  [SKIP] Ollama not available -- skipping Financial Ratio benchmark")
        return []

    model_names = ["base", "finetuned", "rag", "hybrid"]
    model_fns = {
        "base": call_base_model,
        "finetuned": call_finetuned_model,
        "rag": call_rag_model,
        "hybrid": call_hybrid_model,
    }

    total_cases = len(FINANCIAL_RATIO_CASES)
    print(f"  Running {total_cases} financial ratio cases x {len(model_names)} approaches ({LLM_MODEL})...")
    results = []
    start_time = time.perf_counter()

    for i, case in enumerate(FINANCIAL_RATIO_CASES):
        q, table, ctx = case["question"], case["table"], case["context"]
        expected = case["expected"]

        row = {"id": case["id"], "question": q, "expected": expected,
               "category": case["category"]}

        for name in model_names:
            try:
                r = model_fns[name](q, table, ctx)
                correct, extracted = _check_numerical(r.answer, expected)
                row[f"{name}_answer"] = r.answer[:500]
                row[f"{name}_extracted"] = extracted
                row[f"{name}_correct"] = correct
                row[f"{name}_latency_ms"] = r.latency_ms
            except Exception as e:
                row[f"{name}_correct"] = False
                row[f"{name}_answer"] = str(e)[:100]

        results.append(row)

        # Progress with ETA
        elapsed = time.perf_counter() - start_time
        avg_per_case = elapsed / (i + 1)
        remaining = total_cases - (i + 1)
        eta = avg_per_case * remaining

        marks = "  ".join(
            f"{n}={'Y' if row.get(f'{n}_correct') else 'N'}"
            for n in model_names
        )
        print(f"    [{i+1:>2}/{total_cases}] expected={expected:>6}  {marks}  "
              f"(elapsed: {_format_time(elapsed)}, ETA: {_format_time(eta)})")

        # Incremental save after each case
        _save_financial_ratio_results(results, model_names)
        print(f"           -> Saved {i+1}/{total_cases} results to {RESULTS_PATH.name}")

    total_elapsed = time.perf_counter() - start_time
    print(f"\n  Financial Ratio benchmark complete in {_format_time(total_elapsed)}")
    return results


# -------------------------------------------------------------------------
# Streaming (case-by-case) versions for live UI
# -------------------------------------------------------------------------
SENTIMENT_MODEL_NAMES = ["base", "finbert", "rag", "hybrid"]
NUMERICAL_MODEL_NAMES = ["base", "finetuned", "rag", "hybrid"]
FINANCIAL_RATIO_MODEL_NAMES = ["base", "finetuned", "rag", "hybrid"]


def get_sentiment_cases():
    """Return the sentiment test cases list."""
    with open(TEST_CASES_PATH) as f:
        return json.load(f)["sentiment"]


def get_numerical_cases():
    """Return the numerical reasoning test cases list."""
    return NUMERICAL_CASES


def get_financial_ratio_cases():
    """Return the financial ratio test cases list."""
    return FINANCIAL_RATIO_CASES


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
        row[f"{model_name}_answer"] = r.answer[:500]
        row[f"{model_name}_extracted"] = extracted
        row[f"{model_name}_correct"] = correct
        row[f"{model_name}_latency_ms"] = r.latency_ms
        return {"answer": r.answer[:500], "correct": correct,
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
    """Run all three benchmark sections and save."""
    print("=" * 70)
    print("BENCHMARK: Three controlled experiments")
    print("=" * 70)

    # Section 1: BERT 110M
    print("\n--- Section 1: BERT 110M (Sentiment) ---")
    sent_results = run_bert_sentiment_benchmark()
    sent_summary = compute_section_summary(
        sent_results, ["base", "finbert", "rag", "hybrid"])

    # Section 2: Llama2 7B Numerical
    print("\n--- Section 2: Llama2 7B (Numerical Reasoning) ---")
    num_results = run_llama_numerical_benchmark()
    num_summary = compute_section_summary(
        num_results, ["base", "finetuned", "rag", "hybrid"])

    # Section 3: Llama2 7B Financial Ratios
    print("\n--- Section 3: Llama2 7B (Financial Ratios) ---")
    ratio_results = run_financial_ratio_benchmark()
    ratio_summary = compute_section_summary(
        ratio_results, ["base", "finetuned", "rag", "hybrid"])

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
            "llama2_7b_financial_ratios": {
                "title": "Llama2 7B -- Financial Ratios",
                "architecture": "Llama2-7B (7B parameters)",
                "models": ["base", "finetuned", "rag", "hybrid"],
                "model_labels": {
                    "base": "Base Llama2-7B",
                    "finetuned": "FinQA-7B (fine-tuned Llama2-7B)",
                    "rag": "Llama2-7B + RAG (base model)",
                    "hybrid": "FinQA-7B + RAG (fine-tuned + retrieval)",
                },
                "summary": ratio_summary,
                "results": ratio_results,
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
