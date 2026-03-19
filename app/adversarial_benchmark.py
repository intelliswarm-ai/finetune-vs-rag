"""
Adversarial Stress Test Benchmark Runner

Runs all 4 experiments against adversarial test cases:
  - 30 adversarial sentiment cases (BERT 110M)
  - 30 adversarial numerical reasoning cases (Llama2 7B)
  - 30 adversarial financial ratio cases (Llama2 7B)
  - 30 adversarial spam detection cases (DistilBERT 66M)

Each case has an adversarial_type from: noisy_retrieval, knowledge_conflict,
out_of_distribution.

Includes optional LLM-as-Judge evaluation for every model response.

Usage:
    python app/adversarial_benchmark.py
    python app/adversarial_benchmark.py --with-judge
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark import (
    estimate_cost, estimate_tokens_from_text,
    _check_numerical, _extract_number, _format_time,
    compute_section_summary, MARKET_PRICES,
    SPAM_LABELS, FINANCIAL_RATIO_LABELS,
)

ADVERSARIAL_RESULTS_PATH = Path(__file__).parent.parent / "data" / "adversarial_results.json"
ADVERSARIAL_CASES_PATH = Path(__file__).parent.parent / "data" / "adversarial_test_cases.json"


ADVERSARIAL_SENTIMENT_LABELS = {
    "base": "Base BERT (no FT, no RAG)",
    "finbert": "FinBERT (fine-tuned)",
    "rag": "BERT + RAG (retrieval + voting)",
    "hybrid": "FinBERT + RAG (hybrid)",
}

ADVERSARIAL_NUMERICAL_LABELS = {
    "base": "Base Llama2-7B",
    "finetuned": "FinQA-7B (fine-tuned Llama2-7B)",
    "rag": "Llama2-7B + RAG (base model)",
    "hybrid": "FinQA-7B + RAG (fine-tuned + retrieval)",
}

ADVERSARIAL_SPAM_LABELS = {
    "base": "Base DistilBERT (no FT, no RAG)",
    "finetuned": "Fine-tuned DistilBERT (spam-trained)",
    "rag": "DistilBERT + RAG (retrieval + voting)",
    "hybrid": "Fine-tuned DistilBERT + RAG (hybrid)",
}


def _load_adversarial_cases():
    """Load all adversarial test cases."""
    with open(ADVERSARIAL_CASES_PATH) as f:
        return json.load(f)


# -------------------------------------------------------------------------
# Section 1: Adversarial Sentiment (BERT 110M)
# -------------------------------------------------------------------------
def run_adversarial_sentiment(with_judge=False):
    """Run adversarial sentiment benchmark."""
    from demo_utils import (
        run_finbert, run_base_bert_sentiment, run_rag_sentiment,
        run_hybrid_sentiment,
    )

    cases = _load_adversarial_cases()["adversarial_sentiment"]
    model_names = ["base", "finbert", "rag", "hybrid"]
    print(f"  Running {len(cases)} adversarial sentiment cases x {len(model_names)} models...")
    results = []

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name, judge_sentiment, judge_score_to_dict
        judge_model = get_judge_model_name()
        if judge_model:
            print(f"  LLM Judge: {judge_model}")
        else:
            print("  [WARN] No judge model available, skipping judge evaluation")
            with_judge = False

    for i, case in enumerate(cases):
        text, expected = case["text"], case["label"]
        row = {
            "id": case["id"], "text": text, "expected": expected,
            "category": case["category"],
            "adversarial_type": case["adversarial_type"],
        }

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
                row[f"{name}_input_tokens"] = r.input_tokens
                row[f"{name}_completion_tokens"] = 0
                row[f"{name}_total_tokens"] = r.input_tokens
                row[f"{name}_cost_usd"] = estimate_cost(r.input_tokens, 0, "bert_110m")

                if with_judge:
                    score = judge_sentiment(text, expected, r.label, r.confidence, judge_model)
                    if score:
                        row[f"{name}_judge"] = judge_score_to_dict(score)
            except Exception:
                row[f"{name}_label"] = "error"
                row[f"{name}_correct"] = False

        results.append(row)
        marks = "  ".join(
            f"{n}={row.get(f'{n}_label','?'):>8}[{'Y' if row.get(f'{n}_correct') else 'N'}]"
            for n in model_names
        )
        print(f"    [{i+1:>2}/{len(cases)}] exp={expected:>8}  {marks}  [{case['category']}]")

    return results


# -------------------------------------------------------------------------
# Section 2: Adversarial Numerical Reasoning (Llama2 7B)
# -------------------------------------------------------------------------
def run_adversarial_numerical(with_judge=False):
    """Run adversarial numerical reasoning benchmark."""
    from demo_utils import (
        call_finetuned_model, call_base_model, call_rag_model,
        call_hybrid_model, has_llm, LLM_MODEL,
    )

    if not has_llm():
        print("  [SKIP] Ollama not available -- skipping adversarial numerical")
        return []

    cases = _load_adversarial_cases()["adversarial_numerical"]
    model_names = ["base", "finetuned", "rag", "hybrid"]
    model_fns = {
        "base": call_base_model,
        "finetuned": call_finetuned_model,
        "rag": call_rag_model,
        "hybrid": call_hybrid_model,
    }

    print(f"  Running {len(cases)} adversarial numerical cases x {len(model_names)} approaches ({LLM_MODEL})...")
    results = []
    start_time = time.perf_counter()

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name, judge_numerical, judge_score_to_dict
        judge_model = get_judge_model_name()
        if judge_model:
            print(f"  LLM Judge: {judge_model}")
        else:
            print("  [WARN] No judge model available, skipping judge evaluation")
            with_judge = False

    for i, case in enumerate(cases):
        q, table, ctx = case["question"], case["table"], case["context"]
        expected = case["expected"]

        row = {
            "id": case["id"], "question": q, "expected": expected,
            "category": case["category"],
            "adversarial_type": case["adversarial_type"],
        }

        for name in model_names:
            try:
                r = model_fns[name](q, table, ctx)
                correct, extracted = _check_numerical(r.answer, expected)
                row[f"{name}_answer"] = r.answer[:500]
                row[f"{name}_extracted"] = extracted
                row[f"{name}_correct"] = correct
                row[f"{name}_latency_ms"] = r.latency_ms
                row[f"{name}_prompt_tokens"] = r.prompt_tokens
                row[f"{name}_completion_tokens"] = r.completion_tokens
                row[f"{name}_total_tokens"] = r.total_tokens
                row[f"{name}_cost_usd"] = estimate_cost(
                    r.prompt_tokens, r.completion_tokens, "llama2_7b")

                if with_judge:
                    from llm_judge import judge_numerical as jn, judge_score_to_dict as jsd
                    score = jn(q, table, ctx, expected, r.answer[:500], judge_model)
                    if score:
                        row[f"{name}_judge"] = jsd(score)
            except Exception as e:
                row[f"{name}_correct"] = False
                row[f"{name}_answer"] = str(e)[:100]

        results.append(row)

        elapsed = time.perf_counter() - start_time
        avg_per_case = elapsed / (i + 1)
        eta = avg_per_case * (len(cases) - (i + 1))

        marks = "  ".join(
            f"{n}={'Y' if row.get(f'{n}_correct') else 'N'}"
            for n in model_names
        )
        print(f"    [{i+1:>2}/{len(cases)}] expected={expected:>10}  {marks}  "
              f"[{case['category']}]  (ETA: {_format_time(eta)})")

    return results


# -------------------------------------------------------------------------
# Section 3: Adversarial Financial Ratios (Llama2 7B)
# -------------------------------------------------------------------------
def run_adversarial_financial_ratios(with_judge=False):
    """Run adversarial financial ratio benchmark."""
    from demo_utils import (
        call_finetuned_model, call_base_model, call_rag_model,
        call_hybrid_model, has_llm, LLM_MODEL,
    )

    if not has_llm():
        print("  [SKIP] Ollama not available -- skipping adversarial financial ratios")
        return []

    cases = _load_adversarial_cases()["adversarial_financial_ratios"]
    model_names = ["base", "finetuned", "rag", "hybrid"]
    model_fns = {
        "base": call_base_model,
        "finetuned": call_finetuned_model,
        "rag": call_rag_model,
        "hybrid": call_hybrid_model,
    }

    print(f"  Running {len(cases)} adversarial financial ratio cases x {len(model_names)} approaches ({LLM_MODEL})...")
    results = []
    start_time = time.perf_counter()

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name, judge_numerical, judge_score_to_dict
        judge_model = get_judge_model_name()
        if judge_model:
            print(f"  LLM Judge: {judge_model}")
        else:
            print("  [WARN] No judge model available, skipping judge evaluation")
            with_judge = False

    for i, case in enumerate(cases):
        q, table, ctx = case["question"], case["table"], case["context"]
        expected = case["expected"]

        row = {
            "id": case["id"], "question": q, "expected": expected,
            "category": case["category"],
            "adversarial_type": case["adversarial_type"],
        }

        for name in model_names:
            try:
                r = model_fns[name](q, table, ctx)
                correct, extracted = _check_numerical(r.answer, expected)
                row[f"{name}_answer"] = r.answer[:500]
                row[f"{name}_extracted"] = extracted
                row[f"{name}_correct"] = correct
                row[f"{name}_latency_ms"] = r.latency_ms
                row[f"{name}_prompt_tokens"] = r.prompt_tokens
                row[f"{name}_completion_tokens"] = r.completion_tokens
                row[f"{name}_total_tokens"] = r.total_tokens
                row[f"{name}_cost_usd"] = estimate_cost(
                    r.prompt_tokens, r.completion_tokens, "llama2_7b")

                if with_judge:
                    from llm_judge import judge_numerical as jn, judge_score_to_dict as jsd
                    score = jn(q, table, ctx, expected, r.answer[:500], judge_model)
                    if score:
                        row[f"{name}_judge"] = jsd(score)
            except Exception as e:
                row[f"{name}_correct"] = False
                row[f"{name}_answer"] = str(e)[:100]

        results.append(row)

        elapsed = time.perf_counter() - start_time
        avg_per_case = elapsed / (i + 1)
        eta = avg_per_case * (len(cases) - (i + 1))

        marks = "  ".join(
            f"{n}={'Y' if row.get(f'{n}_correct') else 'N'}"
            for n in model_names
        )
        print(f"    [{i+1:>2}/{len(cases)}] expected={expected:>10}  {marks}  "
              f"[{case['category']}]  (ETA: {_format_time(eta)})")

    return results


# -------------------------------------------------------------------------
# Section 4: Adversarial Spam Detection (DistilBERT 66M)
# -------------------------------------------------------------------------
def run_adversarial_spam(with_judge=False):
    """Run adversarial spam detection benchmark."""
    from demo_utils import (
        run_base_distilbert_spam, run_finetuned_distilbert_spam,
        run_rag_spam, run_hybrid_spam,
    )

    cases = _load_adversarial_cases()["adversarial_spam"]
    model_names = ["base", "finetuned", "rag", "hybrid"]
    print(f"  Running {len(cases)} adversarial spam cases x {len(model_names)} models...")
    results = []

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name, judge_spam, judge_score_to_dict
        judge_model = get_judge_model_name()
        if judge_model:
            print(f"  LLM Judge: {judge_model}")
        else:
            print("  [WARN] No judge model available, skipping judge evaluation")
            with_judge = False

    for i, case in enumerate(cases):
        text, expected = case["text"], case["label"]
        row = {
            "id": case["id"], "text": text, "expected": expected,
            "category": case["category"],
            "adversarial_type": case["adversarial_type"],
        }

        for name, fn in [("base", run_base_distilbert_spam),
                         ("finetuned", run_finetuned_distilbert_spam),
                         ("rag", run_rag_spam),
                         ("hybrid", run_hybrid_spam)]:
            try:
                r = fn(text)
                row[f"{name}_label"] = r.label
                row[f"{name}_confidence"] = r.confidence
                row[f"{name}_latency_ms"] = r.latency_ms
                row[f"{name}_correct"] = r.label == expected
                row[f"{name}_input_tokens"] = r.input_tokens
                row[f"{name}_completion_tokens"] = 0
                row[f"{name}_total_tokens"] = r.input_tokens
                row[f"{name}_cost_usd"] = estimate_cost(r.input_tokens, 0, "distilbert_66m")

                if with_judge:
                    score = judge_spam(text, expected, r.label, r.confidence, judge_model)
                    if score:
                        row[f"{name}_judge"] = judge_score_to_dict(score)
            except Exception:
                row[f"{name}_label"] = "error"
                row[f"{name}_correct"] = False

        results.append(row)
        marks = "  ".join(
            f"{n}={row.get(f'{n}_label','?'):>4}[{'Y' if row.get(f'{n}_correct') else 'N'}]"
            for n in model_names
        )
        print(f"    [{i+1:>2}/{len(cases)}] exp={expected:>4}  {marks}  [{case['category']}]")

    return results


# -------------------------------------------------------------------------
# Streaming helpers for live UI
# -------------------------------------------------------------------------
SENTIMENT_MODEL_NAMES = ["base", "finbert", "rag", "hybrid"]
NUMERICAL_MODEL_NAMES = ["base", "finetuned", "rag", "hybrid"]
FINANCIAL_RATIO_MODEL_NAMES = ["base", "finetuned", "rag", "hybrid"]
SPAM_MODEL_NAMES = ["base", "finetuned", "rag", "hybrid"]


def get_adversarial_sentiment_cases():
    return _load_adversarial_cases()["adversarial_sentiment"]

def get_adversarial_numerical_cases():
    return _load_adversarial_cases()["adversarial_numerical"]

def get_adversarial_financial_ratio_cases():
    return _load_adversarial_cases()["adversarial_financial_ratios"]

def get_adversarial_spam_cases():
    return _load_adversarial_cases()["adversarial_spam"]


def run_single_adversarial_sentiment_case(case, with_judge=False, judge_model=None):
    """Run all 4 models on a single adversarial sentiment case."""
    from demo_utils import (
        run_finbert, run_base_bert_sentiment, run_rag_sentiment,
        run_hybrid_sentiment,
    )

    text, expected = case["text"], case["label"]
    row = {
        "id": case["id"], "text": text, "expected": expected,
        "category": case["category"],
        "adversarial_type": case["adversarial_type"],
    }

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
            row[f"{name}_input_tokens"] = r.input_tokens
            row[f"{name}_completion_tokens"] = 0
            row[f"{name}_total_tokens"] = r.input_tokens
            row[f"{name}_cost_usd"] = estimate_cost(r.input_tokens, 0, "bert_110m")

            if with_judge and judge_model:
                from llm_judge import judge_sentiment, judge_score_to_dict
                score = judge_sentiment(text, expected, r.label, r.confidence, judge_model)
                if score:
                    row[f"{name}_judge"] = judge_score_to_dict(score)
        except Exception:
            row[f"{name}_label"] = "error"
            row[f"{name}_correct"] = False

    return row


def run_single_adversarial_numerical_case(case, with_judge=False, judge_model=None):
    """Run all 4 models on a single adversarial numerical case."""
    from demo_utils import (
        call_finetuned_model, call_base_model, call_rag_model,
        call_hybrid_model,
    )

    q, table, ctx = case["question"], case["table"], case["context"]
    expected = case["expected"]
    row = {
        "id": case["id"], "question": q, "expected": expected,
        "category": case["category"],
        "adversarial_type": case["adversarial_type"],
    }

    model_fns = {
        "base": call_base_model,
        "finetuned": call_finetuned_model,
        "rag": call_rag_model,
        "hybrid": call_hybrid_model,
    }

    for name in ["base", "finetuned", "rag", "hybrid"]:
        try:
            r = model_fns[name](q, table, ctx)
            correct, extracted = _check_numerical(r.answer, expected)
            row[f"{name}_answer"] = r.answer[:500]
            row[f"{name}_extracted"] = extracted
            row[f"{name}_correct"] = correct
            row[f"{name}_latency_ms"] = r.latency_ms
            row[f"{name}_prompt_tokens"] = r.prompt_tokens
            row[f"{name}_completion_tokens"] = r.completion_tokens
            row[f"{name}_total_tokens"] = r.total_tokens
            row[f"{name}_cost_usd"] = estimate_cost(
                r.prompt_tokens, r.completion_tokens, "llama2_7b")

            if with_judge and judge_model:
                from llm_judge import judge_numerical, judge_score_to_dict
                score = judge_numerical(q, table, ctx, expected, r.answer[:500], judge_model)
                if score:
                    row[f"{name}_judge"] = judge_score_to_dict(score)
        except Exception as e:
            row[f"{name}_correct"] = False
            row[f"{name}_answer"] = str(e)[:100]

    return row


def run_single_adversarial_spam_case(case, with_judge=False, judge_model=None):
    """Run all 4 models on a single adversarial spam case."""
    from demo_utils import (
        run_base_distilbert_spam, run_finetuned_distilbert_spam,
        run_rag_spam, run_hybrid_spam,
    )

    text, expected = case["text"], case["label"]
    row = {
        "id": case["id"], "text": text, "expected": expected,
        "category": case["category"],
        "adversarial_type": case["adversarial_type"],
    }

    for name, fn in [("base", run_base_distilbert_spam),
                     ("finetuned", run_finetuned_distilbert_spam),
                     ("rag", run_rag_spam),
                     ("hybrid", run_hybrid_spam)]:
        try:
            r = fn(text)
            row[f"{name}_label"] = r.label
            row[f"{name}_confidence"] = r.confidence
            row[f"{name}_latency_ms"] = r.latency_ms
            row[f"{name}_correct"] = r.label == expected
            row[f"{name}_input_tokens"] = r.input_tokens
            row[f"{name}_completion_tokens"] = 0
            row[f"{name}_total_tokens"] = r.input_tokens
            row[f"{name}_cost_usd"] = estimate_cost(r.input_tokens, 0, "distilbert_66m")

            if with_judge and judge_model:
                from llm_judge import judge_spam, judge_score_to_dict
                score = judge_spam(text, expected, r.label, r.confidence, judge_model)
                if score:
                    row[f"{name}_judge"] = judge_score_to_dict(score)
        except Exception:
            row[f"{name}_label"] = "error"
            row[f"{name}_correct"] = False

    return row


# -------------------------------------------------------------------------
# Incremental save helpers
# -------------------------------------------------------------------------

# Section metadata (static config per section)
_SECTION_META = {
    "adversarial_sentiment": {
        "title": "Adversarial Sentiment Classification",
        "architecture": "BERT-base-uncased (110M parameters)",
        "models": ["base", "finbert", "rag", "hybrid"],
        "model_labels": ADVERSARIAL_SENTIMENT_LABELS,
    },
    "adversarial_numerical": {
        "title": "Adversarial Numerical Reasoning",
        "architecture": "Llama2-7B (7B parameters)",
        "models": ["base", "finetuned", "rag", "hybrid"],
        "model_labels": ADVERSARIAL_NUMERICAL_LABELS,
    },
    "adversarial_financial_ratios": {
        "title": "Adversarial Financial Ratios",
        "architecture": "Llama2-7B (7B parameters)",
        "models": ["base", "finetuned", "rag", "hybrid"],
        "model_labels": ADVERSARIAL_NUMERICAL_LABELS,
    },
    "adversarial_spam": {
        "title": "Adversarial Spam Detection",
        "architecture": "DistilBERT-base-uncased (66M parameters)",
        "models": ["base", "finetuned", "rag", "hybrid"],
        "model_labels": ADVERSARIAL_SPAM_LABELS,
    },
}


def _save_incremental(all_results, with_judge):
    """Save current state of all results to disk immediately.

    Called after every single case so nothing is lost on crash.
    """
    from llm_judge import compute_judge_summary

    judge_summaries = {}
    sections = {}

    for section_key, meta in _SECTION_META.items():
        results = all_results.get(section_key, [])
        summary = compute_section_summary(results, meta["models"]) if results else {}
        sections[section_key] = {
            **meta,
            "summary": summary,
            "results": results,
        }
        if with_judge and results:
            judge_summaries[section_key] = compute_judge_summary(
                results, meta["models"])

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_type": "adversarial_stress_test",
        "with_judge": with_judge,
        "judge_summaries": judge_summaries,
        "sections": sections,
    }

    ADVERSARIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ADVERSARIAL_RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)


# -------------------------------------------------------------------------
# Full benchmark
# -------------------------------------------------------------------------
def run_full_adversarial_benchmark(with_judge=False):
    """Run all four adversarial benchmark sections, saving after each case."""
    print("=" * 70)
    print("ADVERSARIAL STRESS TEST: Four experiments, 30 cases each")
    print("  (results saved incrementally after every case)")
    print("=" * 70)

    if with_judge:
        from llm_judge import get_judge_model_name
        jm = get_judge_model_name()
        if jm:
            print(f"LLM-as-Judge enabled: {jm}")
        else:
            print("[WARN] No judge model found, running without judge")
            with_judge = False

    # Shared state -- accumulates results across all sections
    all_results = {
        "adversarial_sentiment": [],
        "adversarial_numerical": [],
        "adversarial_financial_ratios": [],
        "adversarial_spam": [],
    }

    # Try to resume from existing partial results
    if ADVERSARIAL_RESULTS_PATH.exists():
        try:
            with open(ADVERSARIAL_RESULTS_PATH) as f:
                existing = json.load(f)
            for key in all_results:
                sect = existing.get("sections", {}).get(key, {})
                all_results[key] = sect.get("results", [])
            completed_ids = {
                key: {r["id"] for r in results}
                for key, results in all_results.items()
            }
            total_existing = sum(len(v) for v in all_results.values())
            if total_existing > 0:
                print(f"  Resuming from {total_existing} previously completed cases")
        except Exception:
            completed_ids = {key: set() for key in all_results}
    else:
        completed_ids = {key: set() for key in all_results}

    def save_callback():
        _save_incremental(all_results, with_judge)

    # Section 1: Adversarial Sentiment
    print("\n--- Section 1: Adversarial Sentiment (BERT 110M) ---")
    _run_section_sentiment(all_results, completed_ids, with_judge, save_callback)

    # Section 2: Adversarial Numerical
    print("\n--- Section 2: Adversarial Numerical (Llama2 7B) ---")
    _run_section_numerical(all_results, completed_ids, with_judge, save_callback)

    # Section 3: Adversarial Financial Ratios
    print("\n--- Section 3: Adversarial Financial Ratios (Llama2 7B) ---")
    _run_section_financial_ratios(all_results, completed_ids, with_judge, save_callback)

    # Section 4: Adversarial Spam
    print("\n--- Section 4: Adversarial Spam Detection (DistilBERT 66M) ---")
    _run_section_spam(all_results, completed_ids, with_judge, save_callback)

    # Final save
    save_callback()

    # Print summaries
    from llm_judge import compute_judge_summary
    for section_key, meta in _SECTION_META.items():
        results = all_results[section_key]
        s = compute_section_summary(results, meta["models"]) if results else {}
        labels = meta["model_labels"]
        print(f"\n{'='*60}")
        print(f"ADVERSARIAL: {meta['title']} ({meta['architecture']})")
        print(f"{'='*60}")
        for m in meta["models"]:
            ms = s.get(m, {})
            print(f"  {labels.get(m,m):>40}: {ms.get('accuracy',0):>5}% "
                  f"({ms.get('correct',0)}/{ms.get('total',0)})")

        if with_judge and results:
            js = compute_judge_summary(results, meta["models"])
            print(f"\n  LLM Judge Scores (avg):")
            for m in meta["models"]:
                jm = js.get(m, {})
                if jm.get("count", 0) > 0:
                    print(f"    {labels.get(m,m):>38}: "
                          f"C={jm['correctness']:.1f} R={jm['reasoning_quality']:.1f} "
                          f"F={jm['faithfulness']:.1f} O={jm['overall']:.1f}")

    print(f"\nAdversarial results saved to {ADVERSARIAL_RESULTS_PATH}")
    return all_results


# -------------------------------------------------------------------------
# Section runners with incremental save
# -------------------------------------------------------------------------

def _run_section_sentiment(all_results, completed_ids, with_judge, save_callback):
    from demo_utils import (
        run_finbert, run_base_bert_sentiment, run_rag_sentiment,
        run_hybrid_sentiment,
    )
    cases = _load_adversarial_cases()["adversarial_sentiment"]
    model_names = ["base", "finbert", "rag", "hybrid"]
    skip_ids = completed_ids.get("adversarial_sentiment", set())
    remaining = [c for c in cases if c["id"] not in skip_ids]
    print(f"  Running {len(remaining)} adversarial sentiment cases x {len(model_names)} models "
          f"({len(skip_ids)} already done)...")

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name, judge_sentiment, judge_score_to_dict
        judge_model = get_judge_model_name()
        if judge_model:
            print(f"  LLM Judge: {judge_model}")
        else:
            print("  [WARN] No judge model available, skipping judge evaluation")
            with_judge = False

    for i, case in enumerate(remaining):
        text, expected = case["text"], case["label"]
        row = {
            "id": case["id"], "text": text, "expected": expected,
            "category": case["category"],
            "adversarial_type": case["adversarial_type"],
        }
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
                row[f"{name}_input_tokens"] = r.input_tokens
                row[f"{name}_completion_tokens"] = 0
                row[f"{name}_total_tokens"] = r.input_tokens
                row[f"{name}_cost_usd"] = estimate_cost(r.input_tokens, 0, "bert_110m")
                if with_judge:
                    score = judge_sentiment(text, expected, r.label, r.confidence, judge_model)
                    if score:
                        row[f"{name}_judge"] = judge_score_to_dict(score)
            except Exception:
                row[f"{name}_label"] = "error"
                row[f"{name}_correct"] = False

        all_results["adversarial_sentiment"].append(row)
        save_callback()

        marks = "  ".join(
            f"{n}={row.get(f'{n}_label','?'):>8}[{'Y' if row.get(f'{n}_correct') else 'N'}]"
            for n in model_names
        )
        done = len(skip_ids) + i + 1
        print(f"    [{done:>2}/{len(cases)}] exp={expected:>8}  {marks}  [{case['category']}]  (saved)")


def _run_section_numerical(all_results, completed_ids, with_judge, save_callback):
    from demo_utils import (
        call_finetuned_model, call_base_model, call_rag_model,
        call_hybrid_model, has_llm, LLM_MODEL,
    )
    if not has_llm():
        print("  [SKIP] Ollama not available -- skipping adversarial numerical")
        return

    cases = _load_adversarial_cases()["adversarial_numerical"]
    model_names = ["base", "finetuned", "rag", "hybrid"]
    model_fns = {
        "base": call_base_model, "finetuned": call_finetuned_model,
        "rag": call_rag_model, "hybrid": call_hybrid_model,
    }
    skip_ids = completed_ids.get("adversarial_numerical", set())
    remaining = [c for c in cases if c["id"] not in skip_ids]
    print(f"  Running {len(remaining)} adversarial numerical cases x {len(model_names)} approaches ({LLM_MODEL})  "
          f"({len(skip_ids)} already done)...")
    start_time = time.perf_counter()

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name, judge_numerical, judge_score_to_dict
        judge_model = get_judge_model_name()
        if judge_model:
            print(f"  LLM Judge: {judge_model}")
        else:
            print("  [WARN] No judge model available, skipping judge evaluation")
            with_judge = False

    for i, case in enumerate(remaining):
        q, table, ctx = case["question"], case["table"], case["context"]
        expected = case["expected"]
        row = {
            "id": case["id"], "question": q, "expected": expected,
            "category": case["category"],
            "adversarial_type": case["adversarial_type"],
        }
        for name in model_names:
            try:
                r = model_fns[name](q, table, ctx)
                correct, extracted = _check_numerical(r.answer, expected)
                row[f"{name}_answer"] = r.answer[:500]
                row[f"{name}_extracted"] = extracted
                row[f"{name}_correct"] = correct
                row[f"{name}_latency_ms"] = r.latency_ms
                row[f"{name}_prompt_tokens"] = r.prompt_tokens
                row[f"{name}_completion_tokens"] = r.completion_tokens
                row[f"{name}_total_tokens"] = r.total_tokens
                row[f"{name}_cost_usd"] = estimate_cost(
                    r.prompt_tokens, r.completion_tokens, "llama2_7b")
                if with_judge:
                    from llm_judge import judge_numerical as jn, judge_score_to_dict as jsd
                    score = jn(q, table, ctx, expected, r.answer[:500], judge_model)
                    if score:
                        row[f"{name}_judge"] = jsd(score)
            except Exception as e:
                row[f"{name}_correct"] = False
                row[f"{name}_answer"] = str(e)[:100]

        all_results["adversarial_numerical"].append(row)
        save_callback()

        elapsed = time.perf_counter() - start_time
        avg_per_case = elapsed / (i + 1)
        eta = avg_per_case * (len(remaining) - (i + 1))
        marks = "  ".join(
            f"{n}={'Y' if row.get(f'{n}_correct') else 'N'}" for n in model_names
        )
        done = len(skip_ids) + i + 1
        print(f"    [{done:>2}/{len(cases)}] expected={expected:>10}  {marks}  "
              f"[{case['category']}]  (saved, ETA: {_format_time(eta)})")


def _run_section_financial_ratios(all_results, completed_ids, with_judge, save_callback):
    from demo_utils import (
        call_finetuned_model, call_base_model, call_rag_model,
        call_hybrid_model, has_llm, LLM_MODEL,
    )
    if not has_llm():
        print("  [SKIP] Ollama not available -- skipping adversarial financial ratios")
        return

    cases = _load_adversarial_cases()["adversarial_financial_ratios"]
    model_names = ["base", "finetuned", "rag", "hybrid"]
    model_fns = {
        "base": call_base_model, "finetuned": call_finetuned_model,
        "rag": call_rag_model, "hybrid": call_hybrid_model,
    }
    skip_ids = completed_ids.get("adversarial_financial_ratios", set())
    remaining = [c for c in cases if c["id"] not in skip_ids]
    print(f"  Running {len(remaining)} adversarial financial ratio cases x {len(model_names)} approaches ({LLM_MODEL})  "
          f"({len(skip_ids)} already done)...")
    start_time = time.perf_counter()

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name, judge_numerical, judge_score_to_dict
        judge_model = get_judge_model_name()
        if judge_model:
            print(f"  LLM Judge: {judge_model}")
        else:
            print("  [WARN] No judge model available, skipping judge evaluation")
            with_judge = False

    for i, case in enumerate(remaining):
        q, table, ctx = case["question"], case["table"], case["context"]
        expected = case["expected"]
        row = {
            "id": case["id"], "question": q, "expected": expected,
            "category": case["category"],
            "adversarial_type": case["adversarial_type"],
        }
        for name in model_names:
            try:
                r = model_fns[name](q, table, ctx)
                correct, extracted = _check_numerical(r.answer, expected)
                row[f"{name}_answer"] = r.answer[:500]
                row[f"{name}_extracted"] = extracted
                row[f"{name}_correct"] = correct
                row[f"{name}_latency_ms"] = r.latency_ms
                row[f"{name}_prompt_tokens"] = r.prompt_tokens
                row[f"{name}_completion_tokens"] = r.completion_tokens
                row[f"{name}_total_tokens"] = r.total_tokens
                row[f"{name}_cost_usd"] = estimate_cost(
                    r.prompt_tokens, r.completion_tokens, "llama2_7b")
                if with_judge:
                    from llm_judge import judge_numerical as jn, judge_score_to_dict as jsd
                    score = jn(q, table, ctx, expected, r.answer[:500], judge_model)
                    if score:
                        row[f"{name}_judge"] = jsd(score)
            except Exception as e:
                row[f"{name}_correct"] = False
                row[f"{name}_answer"] = str(e)[:100]

        all_results["adversarial_financial_ratios"].append(row)
        save_callback()

        elapsed = time.perf_counter() - start_time
        avg_per_case = elapsed / (i + 1)
        eta = avg_per_case * (len(remaining) - (i + 1))
        marks = "  ".join(
            f"{n}={'Y' if row.get(f'{n}_correct') else 'N'}" for n in model_names
        )
        done = len(skip_ids) + i + 1
        print(f"    [{done:>2}/{len(cases)}] expected={expected:>10}  {marks}  "
              f"[{case['category']}]  (saved, ETA: {_format_time(eta)})")


def _run_section_spam(all_results, completed_ids, with_judge, save_callback):
    from demo_utils import (
        run_base_distilbert_spam, run_finetuned_distilbert_spam,
        run_rag_spam, run_hybrid_spam,
    )
    cases = _load_adversarial_cases()["adversarial_spam"]
    model_names = ["base", "finetuned", "rag", "hybrid"]
    skip_ids = completed_ids.get("adversarial_spam", set())
    remaining = [c for c in cases if c["id"] not in skip_ids]
    print(f"  Running {len(remaining)} adversarial spam cases x {len(model_names)} models "
          f"({len(skip_ids)} already done)...")

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name, judge_spam, judge_score_to_dict
        judge_model = get_judge_model_name()
        if judge_model:
            print(f"  LLM Judge: {judge_model}")
        else:
            print("  [WARN] No judge model available, skipping judge evaluation")
            with_judge = False

    for i, case in enumerate(remaining):
        text, expected = case["text"], case["label"]
        row = {
            "id": case["id"], "text": text, "expected": expected,
            "category": case["category"],
            "adversarial_type": case["adversarial_type"],
        }
        for name, fn in [("base", run_base_distilbert_spam),
                         ("finetuned", run_finetuned_distilbert_spam),
                         ("rag", run_rag_spam),
                         ("hybrid", run_hybrid_spam)]:
            try:
                r = fn(text)
                row[f"{name}_label"] = r.label
                row[f"{name}_confidence"] = r.confidence
                row[f"{name}_latency_ms"] = r.latency_ms
                row[f"{name}_correct"] = r.label == expected
                row[f"{name}_input_tokens"] = r.input_tokens
                row[f"{name}_completion_tokens"] = 0
                row[f"{name}_total_tokens"] = r.input_tokens
                row[f"{name}_cost_usd"] = estimate_cost(r.input_tokens, 0, "distilbert_66m")
                if with_judge:
                    score = judge_spam(text, expected, r.label, r.confidence, judge_model)
                    if score:
                        row[f"{name}_judge"] = judge_score_to_dict(score)
            except Exception:
                row[f"{name}_label"] = "error"
                row[f"{name}_correct"] = False

        all_results["adversarial_spam"].append(row)
        save_callback()

        marks = "  ".join(
            f"{n}={row.get(f'{n}_label','?'):>4}[{'Y' if row.get(f'{n}_correct') else 'N'}]"
            for n in model_names
        )
        done = len(skip_ids) + i + 1
        print(f"    [{done:>2}/{len(cases)}] exp={expected:>4}  {marks}  [{case['category']}]  (saved)")


if __name__ == "__main__":
    with_judge = "--with-judge" in sys.argv
    run_full_adversarial_benchmark(with_judge=with_judge)
