"""
RAG Strengths Benchmark Runner

30 test cases designed to evaluate RAG on its structural advantages:
  - Direct Retrieval: specific facts from Meridian documents
  - Formula + Aligned Data: formulas applied to Meridian's own numbers
  - Cross-Document Synthesis: combining info from multiple documents
  - Contextual Interpretation: understanding document narrative
  - Trend Analysis: tracking metrics over time

All questions reference Meridian National Bancorp data that exists in
the RAG knowledge base, so RAG retrieval should help (not hurt).

Includes LLM-as-Judge evaluation for every model response.

Usage:
    python app/rag_strengths_benchmark.py
    python app/rag_strengths_benchmark.py --with-judge
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark import (
    estimate_cost, estimate_tokens_from_text,
    _format_time, compute_section_summary,
)

RAG_STRENGTHS_RESULTS_PATH = Path(__file__).parent.parent / "data" / "rag_strengths_results.json"
RAG_STRENGTHS_CASES_PATH = Path(__file__).parent.parent / "data" / "rag_strengths_benchmark.json"


MODEL_LABELS = {
    "base": "Base Llama2-7B (no RAG)",
    "rag": "Llama2-7B + RAG",
    "finetuned": "FinQA-7B (no RAG)",
    "hybrid": "FinQA-7B + RAG (hybrid)",
}

MODEL_NAMES = ["base", "rag", "finetuned", "hybrid"]

# Category display names
CATEGORY_NAMES = {
    "direct_retrieval": "Direct Retrieval",
    "formula_with_aligned_data": "Formula + Aligned Data",
    "cross_document_synthesis": "Cross-Document Synthesis",
    "contextual_interpretation": "Contextual Interpretation",
    "trend_analysis": "Trend Analysis",
}


def _load_cases():
    """Load all RAG strengths test cases, flattened into a single list."""
    with open(RAG_STRENGTHS_CASES_PATH) as f:
        data = json.load(f)

    cases = []
    for category_key in ["direct_retrieval", "formula_with_aligned_data",
                         "cross_document_synthesis", "contextual_interpretation",
                         "trend_analysis"]:
        for case in data.get(category_key, []):
            case["benchmark_category"] = category_key
            # Normalize source_document(s) to a list
            sources = case.get("source_documents", case.get("source_document", ""))
            if isinstance(sources, str):
                sources = [sources] if sources else []
            case["source_documents"] = sources
            cases.append(case)

    return cases


def _check_answer(model_answer: str, expected: str) -> bool:
    """Check if model's answer contains the expected value(s).

    Handles multiple expected values separated by commas, and checks for
    both numeric and string matches.
    """
    if not model_answer:
        return False

    answer_lower = model_answer.lower()

    # Split expected into parts (comma-separated for multi-fact answers)
    # Only split on commas that separate distinct facts
    expected_parts = [p.strip() for p in expected.split(",") if p.strip()]

    # For multi-part answers, require at least the first (primary) value
    primary = expected_parts[0] if expected_parts else expected

    # Try to extract numbers from the primary expected value
    nums = re.findall(r'[\-]?\d+\.?\d*', primary)
    if nums:
        # Check if any expected number appears in the answer
        for num_str in nums:
            try:
                exp_val = float(num_str)
                # Find all numbers in the model answer
                answer_nums = re.findall(r'[\-]?\d+\.?\d*', model_answer)
                for ans_str in answer_nums:
                    try:
                        ans_val = float(ans_str)
                        if exp_val == 0:
                            if abs(ans_val) < 0.1:
                                return True
                        elif abs(ans_val - exp_val) / abs(exp_val) <= 0.05:
                            return True
                    except ValueError:
                        continue
            except ValueError:
                continue

    # Fallback: check if key parts of expected appear in answer (case-insensitive)
    # This handles string answers like "Consumer Banking grew fastest"
    key_terms = re.findall(r'[A-Za-z]{4,}', primary)
    if key_terms and len(key_terms) >= 2:
        matches = sum(1 for t in key_terms if t.lower() in answer_lower)
        if matches >= len(key_terms) * 0.5:
            return True

    return False


def get_cases():
    """Public accessor for test cases."""
    return _load_cases()


# -------------------------------------------------------------------------
# Section meta for result structure
# -------------------------------------------------------------------------
_SECTION_META = {
    "rag_strengths": {
        "title": "RAG Strengths Benchmark",
        "architecture": "Llama2-7B (7B parameters)",
        "models": MODEL_NAMES,
        "model_labels": MODEL_LABELS,
    },
}


def run_single_case(case, with_judge=False, judge_model=None):
    """Run all models on a single RAG strengths case."""
    from demo_utils import (
        call_finetuned_model, call_base_model, call_rag_model,
        call_hybrid_model,
    )

    question = case["question"]
    expected = case["expected"]
    category = case["benchmark_category"]
    source_docs = ", ".join(case.get("source_documents", []))

    row = {
        "id": case["id"],
        "question": question,
        "expected": expected,
        "category": category,
        "difficulty": case.get("difficulty", "medium"),
        "source_documents": source_docs,
        "why_rag_wins": case.get("why_rag_wins", ""),
    }

    model_fns = {
        "base": call_base_model,
        "rag": call_rag_model,
        "finetuned": call_finetuned_model,
        "hybrid": call_hybrid_model,
    }

    for name in MODEL_NAMES:
        try:
            # No table or context -- the model must rely on its knowledge
            # or RAG retrieval to answer questions about Meridian
            r = model_fns[name](question, "", "")
            answer_text = r.answer[:800] if r.answer else ""
            correct = _check_answer(answer_text, expected)

            row[f"{name}_answer"] = answer_text
            row[f"{name}_correct"] = correct
            row[f"{name}_latency_ms"] = r.latency_ms
            row[f"{name}_prompt_tokens"] = r.prompt_tokens
            row[f"{name}_completion_tokens"] = r.completion_tokens
            row[f"{name}_total_tokens"] = r.total_tokens
            row[f"{name}_cost_usd"] = estimate_cost(
                r.prompt_tokens, r.completion_tokens, "llama2_7b")

            if with_judge and judge_model:
                from llm_judge import judge_retrieval_qa, judge_score_to_dict
                score = judge_retrieval_qa(
                    question, expected, answer_text,
                    source_documents=source_docs,
                    category=category,
                    judge_model=judge_model,
                )
                if score:
                    row[f"{name}_judge"] = judge_score_to_dict(score)
        except Exception as e:
            row[f"{name}_correct"] = False
            row[f"{name}_answer"] = str(e)[:200]

    return row


# -------------------------------------------------------------------------
# Incremental save
# -------------------------------------------------------------------------
def _save_incremental(results, with_judge):
    """Save current state to disk immediately."""
    from llm_judge import compute_judge_summary

    summary = compute_section_summary(results, MODEL_NAMES) if results else {}

    judge_summaries = {}
    if with_judge and results:
        judge_summaries["rag_strengths"] = compute_judge_summary(results, MODEL_NAMES)

    # Category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0}
            for m in MODEL_NAMES:
                categories[cat][f"{m}_correct"] = 0
        categories[cat]["total"] += 1
        for m in MODEL_NAMES:
            if r.get(f"{m}_correct"):
                categories[cat][f"{m}_correct"] += 1

    for cat, data in categories.items():
        total = data["total"]
        for m in MODEL_NAMES:
            correct = data[f"{m}_correct"]
            summary[f"category_{cat}"] = summary.get(f"category_{cat}", {"total": total})
            summary[f"category_{cat}"][f"{m}_accuracy"] = round(
                correct / total * 100, 1) if total > 0 else 0
            summary[f"category_{cat}"]["total"] = total

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_type": "rag_strengths",
        "with_judge": with_judge,
        "judge_summaries": judge_summaries,
        "sections": {
            "rag_strengths": {
                **_SECTION_META["rag_strengths"],
                "summary": summary,
                "results": results,
            },
        },
    }

    RAG_STRENGTHS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAG_STRENGTHS_RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)


# -------------------------------------------------------------------------
# Full benchmark
# -------------------------------------------------------------------------
def run_full_benchmark(with_judge=False):
    """Run the full RAG strengths benchmark."""
    print("=" * 70)
    print("RAG STRENGTHS BENCHMARK: 30 cases across 5 categories")
    print("  (results saved incrementally after every case)")
    print("=" * 70)

    from demo_utils import has_llm, LLM_MODEL

    if not has_llm():
        print("[SKIP] Ollama not available -- cannot run RAG strengths benchmark")
        return []

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name
        judge_model = get_judge_model_name()
        if judge_model:
            print(f"LLM-as-Judge enabled: {judge_model}")
        else:
            print("[WARN] No judge model found, running without judge")
            with_judge = False

    cases = _load_cases()
    results = []

    # Try to resume from existing partial results
    completed_ids = set()
    if RAG_STRENGTHS_RESULTS_PATH.exists():
        try:
            with open(RAG_STRENGTHS_RESULTS_PATH) as f:
                existing = json.load(f)
            sect = existing.get("sections", {}).get("rag_strengths", {})
            results = sect.get("results", [])
            completed_ids = {r["id"] for r in results}
            if completed_ids:
                print(f"  Resuming from {len(completed_ids)} previously completed cases")
        except Exception:
            pass

    remaining = [c for c in cases if c["id"] not in completed_ids]
    print(f"  Running {len(remaining)} cases x {len(MODEL_NAMES)} models ({LLM_MODEL})...")
    start_time = time.perf_counter()

    for i, case in enumerate(remaining):
        row = run_single_case(case, with_judge, judge_model)
        results.append(row)
        _save_incremental(results, with_judge)

        elapsed = time.perf_counter() - start_time
        avg_per_case = elapsed / (i + 1)
        eta = avg_per_case * (len(remaining) - (i + 1))

        marks = "  ".join(
            f"{n}={'Y' if row.get(f'{n}_correct') else 'N'}" for n in MODEL_NAMES
        )
        done = len(completed_ids) + i + 1
        cat = CATEGORY_NAMES.get(case["benchmark_category"], case["benchmark_category"])
        print(f"  [{done:>2}/{len(cases)}] {marks}  [{cat}] {case['id']}  "
              f"(ETA: {_format_time(eta)})")

    # Final save
    _save_incremental(results, with_judge)

    # Print summary
    print(f"\n{'='*60}")
    print("RAG STRENGTHS BENCHMARK RESULTS")
    print(f"{'='*60}")

    for m in MODEL_NAMES:
        correct = sum(1 for r in results if r.get(f"{m}_correct"))
        total = len(results)
        acc = round(correct / total * 100, 1) if total > 0 else 0
        print(f"  {MODEL_LABELS.get(m, m):>35}: {acc:>5}% ({correct}/{total})")

    # Category breakdown
    print(f"\nBy Category:")
    cat_data = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in cat_data:
            cat_data[cat] = {m: {"correct": 0, "total": 0} for m in MODEL_NAMES}
        for m in MODEL_NAMES:
            cat_data[cat][m]["total"] += 1
            if r.get(f"{m}_correct"):
                cat_data[cat][m]["correct"] += 1

    for cat, models in sorted(cat_data.items()):
        cat_name = CATEGORY_NAMES.get(cat, cat)
        print(f"\n  {cat_name}:")
        for m in MODEL_NAMES:
            c = models[m]["correct"]
            t = models[m]["total"]
            acc = round(c / t * 100, 1) if t > 0 else 0
            print(f"    {MODEL_LABELS.get(m, m):>33}: {acc:>5}% ({c}/{t})")

    if with_judge and results:
        from llm_judge import compute_judge_summary
        js = compute_judge_summary(results, MODEL_NAMES)
        print(f"\n  LLM Judge Scores (avg):")
        for m in MODEL_NAMES:
            jm = js.get(m, {})
            if jm.get("count", 0) > 0:
                print(f"    {MODEL_LABELS.get(m, m):>33}: "
                      f"C={jm['correctness']:.1f} R={jm['reasoning_quality']:.1f} "
                      f"F={jm['faithfulness']:.1f} O={jm['overall']:.1f}")

    print(f"\nResults saved to {RAG_STRENGTHS_RESULTS_PATH}")
    return results


if __name__ == "__main__":
    with_judge = "--with-judge" in sys.argv
    run_full_benchmark(with_judge=with_judge)
