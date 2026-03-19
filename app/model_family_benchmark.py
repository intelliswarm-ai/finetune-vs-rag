"""
Model Family Benchmark -- Does a bigger fine-tuned model beat a smaller one?

Compares two fine-tuned models on the SAME spam-detection task:
  - Fine-tuned DistilBERT  (66M parameters, local inference)
  - Fine-tuned GPT-4o-mini (~8B parameters, OpenAI API)

Both were trained on the same spam-detection dataset.
Evaluated on basic test cases (20) and adversarial test cases (30).

Includes optional LLM-as-Judge evaluation for every model response.

Usage:
    python app/model_family_benchmark.py
    python app/model_family_benchmark.py --with-judge
"""
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from benchmark import (
    estimate_cost, estimate_tokens_from_text,
    compute_section_summary, _compute_f1,
    MARKET_PRICES,
)

RESULTS_PATH = Path(__file__).parent.parent / "data" / "model_family_results.json"
BASIC_CASES_PATH = Path(__file__).parent.parent / "data" / "benchmark_test_cases.json"
ADVERSARIAL_CASES_PATH = Path(__file__).parent.parent / "data" / "adversarial_test_cases.json"

# -------------------------------------------------------------------------
# Model configuration
# -------------------------------------------------------------------------
GPT4OMINI_FT_MODEL = "ft:gpt-4o-mini-2024-07-18:personal:spam-detection-llm-gpt-4o-mini:DL8VRkaX"

MODEL_LABELS = {
    "distilbert_ft": "Fine-tuned DistilBERT (66M)",
    "gpt4omini_ft": "Fine-tuned GPT-4o-mini (~8B)",
}

MODEL_INFO = {
    "distilbert_ft": {
        "name": "Fine-tuned DistilBERT",
        "base_model": "distilbert-base-uncased",
        "params": "66M",
        "params_numeric": 66_000_000,
        "type": "Local (HuggingFace)",
        "inference": "Local CPU/GPU",
        "training": "Custom fine-tuning on spam dataset",
    },
    "gpt4omini_ft": {
        "name": "Fine-tuned GPT-4o-mini",
        "base_model": "gpt-4o-mini-2024-07-18",
        "params": "~8B (undisclosed)",
        "params_numeric": 8_000_000_000,
        "type": "Cloud API (OpenAI)",
        "inference": "OpenAI API",
        "training": "OpenAI fine-tuning platform on same spam dataset",
    },
}

# Pricing (USD per 1M tokens)
MARKET_PRICES_FAMILY = {
    "distilbert_ft": {
        "input_per_1m": 0.01,   # local inference, near-zero
        "output_per_1m": 0.00,  # classification, no output tokens
    },
    "gpt4omini_ft": {
        "input_per_1m": 0.30,   # fine-tuned gpt-4o-mini input
        "output_per_1m": 1.20,  # fine-tuned gpt-4o-mini output
    },
}

MODEL_NAMES = ["distilbert_ft", "gpt4omini_ft"]


def _estimate_family_cost(prompt_tokens, completion_tokens, model_key):
    """Estimate cost in USD for model family benchmark."""
    prices = MARKET_PRICES_FAMILY.get(model_key, MARKET_PRICES_FAMILY["distilbert_ft"])
    cost = (prompt_tokens * prices["input_per_1m"] +
            completion_tokens * prices["output_per_1m"]) / 1_000_000
    return round(cost, 8)


# -------------------------------------------------------------------------
# GPT-4o-mini fine-tuned caller
# -------------------------------------------------------------------------
_openai_client = None


def _get_openai_client():
    """Get or create OpenAI client for the fine-tuned GPT-4o-mini model."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set -- required for GPT-4o-mini fine-tuned model")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def has_openai():
    """Check if OpenAI API is available."""
    return bool(os.getenv("OPENAI_API_KEY", ""))


@dataclass
class SpamResult:
    label: str          # "spam" or "ham"
    confidence: float   # 0.0-1.0
    latency_ms: float
    input_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_output: str     # raw model response


def call_gpt4omini_ft(text: str) -> SpamResult:
    """Call the fine-tuned GPT-4o-mini spam detection model via OpenAI API."""
    client = _get_openai_client()
    t0 = time.perf_counter()

    response = client.chat.completions.create(
        model=GPT4OMINI_FT_MODEL,
        messages=[
            {"role": "system", "content": "Classify the following email as 'spam' or 'ham'."},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=10,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    raw = response.choices[0].message.content.strip().lower()

    # Parse label from response
    if "spam" in raw:
        label = "spam"
    elif "ham" in raw:
        label = "ham"
    else:
        label = raw[:10]  # fallback

    # Token usage from API response
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0

    # Confidence: fine-tuned classification models are high-confidence
    # We use logprobs if available, otherwise default to 0.95 for correct format
    confidence = 0.95 if label in ("spam", "ham") else 0.5

    return SpamResult(
        label=label,
        confidence=confidence,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw_output=raw,
    )


def call_distilbert_ft(text: str) -> SpamResult:
    """Call the fine-tuned DistilBERT spam detection model (local)."""
    from demo_utils import run_finetuned_distilbert_spam
    r = run_finetuned_distilbert_spam(text)
    return SpamResult(
        label=r.label,
        confidence=r.confidence,
        latency_ms=r.latency_ms,
        input_tokens=r.input_tokens,
        completion_tokens=0,
        total_tokens=r.input_tokens,
        raw_output=r.label,
    )


# -------------------------------------------------------------------------
# Load test cases
# -------------------------------------------------------------------------
def get_basic_spam_cases():
    """Load basic spam detection test cases."""
    with open(BASIC_CASES_PATH) as f:
        return json.load(f)["spam_detection"]


def get_adversarial_spam_cases():
    """Load adversarial spam detection test cases."""
    with open(ADVERSARIAL_CASES_PATH) as f:
        return json.load(f)["adversarial_spam"]


# -------------------------------------------------------------------------
# Run a single case
# -------------------------------------------------------------------------
MODEL_FNS = {
    "distilbert_ft": call_distilbert_ft,
    "gpt4omini_ft": call_gpt4omini_ft,
}


def run_single_case(case, with_judge=False, judge_model=None):
    """Run both models on a single spam case. Returns result dict."""
    text = case["text"]
    expected = case["label"]
    row = {
        "id": case["id"],
        "text": text,
        "expected": expected,
        "category": case.get("category", ""),
    }
    if "adversarial_type" in case:
        row["adversarial_type"] = case["adversarial_type"]

    for name in MODEL_NAMES:
        try:
            r = MODEL_FNS[name](text)
            row[f"{name}_label"] = r.label
            row[f"{name}_confidence"] = r.confidence
            row[f"{name}_latency_ms"] = r.latency_ms
            row[f"{name}_correct"] = r.label == expected
            row[f"{name}_input_tokens"] = r.input_tokens
            row[f"{name}_completion_tokens"] = r.completion_tokens
            row[f"{name}_total_tokens"] = r.total_tokens
            row[f"{name}_cost_usd"] = _estimate_family_cost(
                r.input_tokens, r.completion_tokens, name)
            row[f"{name}_raw_output"] = r.raw_output

            if with_judge and judge_model:
                from llm_judge import judge_spam, judge_score_to_dict
                score = judge_spam(text, expected, r.label, r.confidence, judge_model)
                if score:
                    row[f"{name}_judge"] = judge_score_to_dict(score)
        except Exception as e:
            row[f"{name}_label"] = "error"
            row[f"{name}_correct"] = False
            row[f"{name}_raw_output"] = str(e)[:200]

    return row


# -------------------------------------------------------------------------
# Section runners
# -------------------------------------------------------------------------
def _format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def run_section(cases, section_name, with_judge=False, judge_model=None):
    """Run both models on a list of cases, returning results list."""
    print(f"  Running {len(cases)} {section_name} cases x {len(MODEL_NAMES)} models...")
    results = []
    start_time = time.perf_counter()

    for i, case in enumerate(cases):
        row = run_single_case(case, with_judge=with_judge, judge_model=judge_model)
        results.append(row)

        elapsed = time.perf_counter() - start_time
        avg_per_case = elapsed / (i + 1)
        eta = avg_per_case * (len(cases) - (i + 1))

        marks = "  ".join(
            f"{n}={row.get(f'{n}_label', '?'):>4}[{'Y' if row.get(f'{n}_correct') else 'N'}]"
            for n in MODEL_NAMES
        )
        print(f"    [{i+1:>2}/{len(cases)}] exp={case['label']:>4}  {marks}  "
              f"[{case.get('category', '')}]  (ETA: {_format_time(eta)})")

        # Incremental save
        _save_results(results, section_name)

    total_elapsed = time.perf_counter() - start_time
    print(f"  {section_name} complete in {_format_time(total_elapsed)}")
    return results


def _save_results(basic_results=None, last_section=None,
                  adversarial_results=None, with_judge=False,
                  judge_summaries=None):
    """Save current results to disk. Preserves existing sections."""
    existing = {}
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    sections = existing.get("sections", {})

    if basic_results is not None:
        basic_summary = compute_section_summary(basic_results, MODEL_NAMES) if basic_results else {}
        sections["basic_spam"] = {
            "title": "Basic Spam Detection -- Model Size Comparison",
            "subtitle": "Same training data, different model sizes",
            "models": list(MODEL_NAMES),
            "model_labels": MODEL_LABELS,
            "model_info": MODEL_INFO,
            "summary": basic_summary,
            "results": basic_results,
        }

    if adversarial_results is not None:
        adv_summary = compute_section_summary(adversarial_results, MODEL_NAMES) if adversarial_results else {}
        sections["adversarial_spam"] = {
            "title": "Adversarial Spam Detection -- Model Size Comparison",
            "subtitle": "Stress testing with adversarial cases",
            "models": list(MODEL_NAMES),
            "model_labels": MODEL_LABELS,
            "model_info": MODEL_INFO,
            "summary": adv_summary,
            "results": adversarial_results,
        }

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_type": "model_family_comparison",
        "description": "Does model size matter? Comparing fine-tuned DistilBERT (66M) vs fine-tuned GPT-4o-mini (~8B) on spam detection.",
        "with_judge": with_judge,
        "judge_summaries": judge_summaries or existing.get("judge_summaries", {}),
        "sections": sections,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = RESULTS_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2)
    tmp_path.replace(RESULTS_PATH)


# Wrapper for incremental save during section run
_current_basic = None
_current_adversarial = None

_orig_save = _save_results


def _save_results(results, section_name, **kwargs):
    """Incremental save wrapper."""
    global _current_basic, _current_adversarial
    if section_name == "basic":
        _current_basic = results
    elif section_name == "adversarial":
        _current_adversarial = results
    _orig_save(
        basic_results=_current_basic,
        adversarial_results=_current_adversarial,
        **kwargs,
    )


# -------------------------------------------------------------------------
# Full benchmark
# -------------------------------------------------------------------------
def run_full_benchmark(with_judge=False):
    """Run both sections of the model family benchmark."""
    global _current_basic, _current_adversarial
    _current_basic = None
    _current_adversarial = None

    print("=" * 70)
    print("MODEL FAMILY BENCHMARK: Does model size matter for fine-tuning?")
    print(f"  DistilBERT (66M) vs GPT-4o-mini (~8B) on spam detection")
    print("=" * 70)

    if not has_openai():
        print("[ERROR] OPENAI_API_KEY not set. Cannot call GPT-4o-mini fine-tuned model.")
        print("Set it in .env or environment and try again.")
        return

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name
        judge_model = get_judge_model_name()
        if judge_model:
            print(f"LLM-as-Judge enabled: {judge_model}")
        else:
            print("[WARN] No judge model found, running without judge")
            with_judge = False

    # Section 1: Basic test cases
    print("\n--- Section 1: Basic Spam Detection (20 cases) ---")
    basic_cases = get_basic_spam_cases()
    basic_results = run_section(basic_cases, "basic",
                                with_judge=with_judge, judge_model=judge_model)

    # Section 2: Adversarial test cases
    print("\n--- Section 2: Adversarial Spam Detection (30 cases) ---")
    adv_cases = get_adversarial_spam_cases()
    adv_results = run_section(adv_cases, "adversarial",
                              with_judge=with_judge, judge_model=judge_model)

    # Compute judge summaries
    judge_summaries = {}
    if with_judge:
        from llm_judge import compute_judge_summary
        if basic_results:
            judge_summaries["basic_spam"] = compute_judge_summary(basic_results, MODEL_NAMES)
        if adv_results:
            judge_summaries["adversarial_spam"] = compute_judge_summary(adv_results, MODEL_NAMES)

    # Final save with judge summaries
    _orig_save(
        basic_results=basic_results,
        adversarial_results=adv_results,
        with_judge=with_judge,
        judge_summaries=judge_summaries,
    )

    # Print summary
    for label, results in [("BASIC", basic_results), ("ADVERSARIAL", adv_results)]:
        if not results:
            continue
        s = compute_section_summary(results, MODEL_NAMES)
        print(f"\n{'='*60}")
        print(f"MODEL FAMILY -- {label} SPAM DETECTION")
        print(f"{'='*60}")
        for m in MODEL_NAMES:
            ms = s.get(m, {})
            print(f"  {MODEL_LABELS[m]:>35}: {ms.get('accuracy', 0):>5}% "
                  f"({ms.get('correct', 0)}/{ms.get('total', 0)})  "
                  f"avg_lat={ms.get('avg_latency_ms', 0):.0f}ms  "
                  f"cost/1K=${ms.get('cost_per_1k_queries_usd', 0):.4f}")

        if with_judge and label.lower() + "_spam" in judge_summaries:
            js = judge_summaries[label.lower() + "_spam"]
            print(f"\n  LLM Judge Scores (avg):")
            for m in MODEL_NAMES:
                jm = js.get(m, {})
                if jm.get("count", 0) > 0:
                    print(f"    {MODEL_LABELS[m]:>33}: "
                          f"C={jm['correctness']:.1f} R={jm['reasoning_quality']:.1f} "
                          f"F={jm['faithfulness']:.1f} O={jm['overall']:.1f}")

    # Size comparison
    print(f"\n{'='*60}")
    print("SIZE COMPARISON")
    print(f"{'='*60}")
    print(f"  DistilBERT:   66M parameters (local, ~$0.01/1M tokens)")
    print(f"  GPT-4o-mini: ~8B parameters (API,   ~$0.30/1M input + $1.20/1M output)")
    print(f"  Size ratio:  ~121x larger")

    if basic_results:
        bs = compute_section_summary(basic_results, MODEL_NAMES)
        db = bs.get("distilbert_ft", {}).get("accuracy", 0)
        gpt = bs.get("gpt4omini_ft", {}).get("accuracy", 0)
        diff = gpt - db
        print(f"\n  Basic accuracy:       DistilBERT={db}%  GPT-4o-mini={gpt}%  (diff={diff:+.1f}%)")

    if adv_results:
        ads = compute_section_summary(adv_results, MODEL_NAMES)
        db = ads.get("distilbert_ft", {}).get("accuracy", 0)
        gpt = ads.get("gpt4omini_ft", {}).get("accuracy", 0)
        diff = gpt - db
        print(f"  Adversarial accuracy: DistilBERT={db}%  GPT-4o-mini={gpt}%  (diff={diff:+.1f}%)")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    with_judge = "--with-judge" in sys.argv
    run_full_benchmark(with_judge=with_judge)
