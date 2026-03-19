"""
Adversarial Stress Test Results Page

Four experiments with 30 adversarial cases each, testing model robustness against:
  - Noisy Retrieval: irrelevant/distracting context
  - Knowledge Conflict: contradictory signals in input
  - Out-of-Distribution: unusual domains or formats

Includes LLM-as-Judge evaluation (optional) with structured scoring:
  Correctness (1-5), Reasoning Quality (1-5), Faithfulness (1-5)
"""
import streamlit as st
import pandas as pd
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Benchmark Adversarial Results", page_icon="FT", layout="wide")

st.title("Benchmark Adversarial Results")
st.markdown("""
**30 adversarial cases per experiment** designed to expose model weaknesses:

| Category | Description | Count per Experiment |
|----------|------------|---------------------|
| **Noisy Retrieval** | Irrelevant context, distracting numbers, red herrings | 10 |
| **Knowledge Conflict** | Contradictory signals, mixed positive/negative indicators | 10 |
| **Out-of-Distribution** | Unusual domains, novel formats, edge-case vocabulary | 10 |

Same four approaches compared: **Base vs Fine-Tuned vs RAG vs Hybrid**.
""")

ADVERSARIAL_RESULTS_PATH = Path(__file__).parent.parent.parent / "data" / "adversarial_results.json"
ADVERSARIAL_CASES_PATH = Path(__file__).parent.parent.parent / "data" / "adversarial_test_cases.json"

COLORS = {"finbert": "#28a745", "finetuned": "#28a745",
          "base": "#dc3545", "rag": "#007bff", "hybrid": "#ff8c00"}

ADV_CATEGORY_COLORS = {
    "noisy_retrieval": "#e67e22",
    "knowledge_conflict": "#e74c3c",
    "out_of_distribution": "#9b59b6",
}

# Load saved results if they exist
adv_results_data = None
if ADVERSARIAL_RESULTS_PATH.exists():
    with open(ADVERSARIAL_RESULTS_PATH) as f:
        adv_results_data = json.load(f)
if adv_results_data and "sections" not in adv_results_data:
    adv_results_data = None


# =========================================================================
# Render a benchmark section (reused pattern from main results page)
# =========================================================================
def render_adversarial_section(section_data, judge_summary=None):
    import plotly.graph_objects as go

    title = section_data["title"]
    arch = section_data["architecture"]
    models = section_data["models"]
    labels = section_data["model_labels"]
    summary = section_data["summary"]
    per_example = section_data["results"]

    st.subheader(f"{title}")
    st.caption(f"Architecture: {arch} -- 30 adversarial cases")

    # Accuracy metrics
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        s = summary.get(m, {})
        with col:
            st.metric(labels.get(m, m),
                       f"{s.get('accuracy', 0)}%",
                       delta=f"{s.get('correct', 0)}/{s.get('total', 0)}")

    # Accuracy chart
    fig = go.Figure()
    for m in models:
        s = summary.get(m, {})
        fig.add_trace(go.Bar(
            name=labels.get(m, m), x=["Accuracy (%)"],
            y=[s.get("accuracy", 0)],
            marker_color=COLORS.get(m, "#999"),
            text=[f"{s.get('accuracy', 0)}%"], textposition="auto",
        ))
    fig.update_layout(title=f"Adversarial Accuracy -- {arch}", barmode="group",
                       yaxis_range=[0, 105])
    st.plotly_chart(fig, use_container_width=True, key=f"adv_acc_{title}")

    # Adversarial category breakdown
    cat_rows = []
    for key, val in sorted(summary.items()):
        if not key.startswith("category_"):
            continue
        cat = key.replace("category_", "").replace("_", " ").title()
        row = {"Adversarial Category": cat, "Cases": val["total"]}
        for m in models:
            row[labels.get(m, m).split("(")[0].strip()] = f"{val.get(f'{m}_accuracy', 0)}%"
        cat_rows.append(row)

    if cat_rows:
        st.markdown("##### Accuracy by Adversarial Category")
        st.table(pd.DataFrame(cat_rows))

        # Category accuracy chart
        categories = [r["Adversarial Category"] for r in cat_rows]
        fig_cat = go.Figure()
        for m in models:
            short = labels.get(m, m).split("(")[0].strip()
            vals = []
            for key, val in sorted(summary.items()):
                if key.startswith("category_"):
                    vals.append(val.get(f"{m}_accuracy", 0))
            fig_cat.add_trace(go.Bar(
                name=short, x=categories, y=vals,
                marker_color=COLORS.get(m, "#999"),
            ))
        fig_cat.update_layout(
            title="Accuracy by Adversarial Category",
            barmode="group", yaxis_range=[0, 105],
        )
        st.plotly_chart(fig_cat, use_container_width=True, key=f"adv_cat_chart_{title}")

    # Latency chart
    lat_data = {m: summary.get(m, {}).get("avg_latency_ms") for m in models}
    if any(v for v in lat_data.values()):
        fig2 = go.Figure()
        for m in models:
            v = lat_data.get(m)
            if v:
                fig2.add_trace(go.Bar(
                    name=labels.get(m, m), x=["Avg Latency (ms)"],
                    y=[v], marker_color=COLORS.get(m, "#999"),
                    text=[f"{v:.0f}ms"], textposition="auto",
                ))
        fig2.update_layout(title="Average Latency", barmode="group")
        st.plotly_chart(fig2, use_container_width=True, key=f"adv_lat_{title}")

    # Confidence chart (classification only)
    conf_data = {m: summary.get(m, {}).get("avg_confidence") for m in models}
    if any(v for v in conf_data.values()):
        fig_conf = go.Figure()
        for m in models:
            v = conf_data.get(m)
            if v:
                fig_conf.add_trace(go.Bar(
                    name=labels.get(m, m), x=["Avg Confidence"],
                    y=[v], marker_color=COLORS.get(m, "#999"),
                    text=[f"{v:.3f}"], textposition="auto",
                ))
        fig_conf.update_layout(title="Average Confidence (Adversarial)", barmode="group",
                                yaxis_range=[0, 1.05])
        st.plotly_chart(fig_conf, use_container_width=True, key=f"adv_conf_{title}")

    # Per-example table with adversarial type
    with st.expander("Per-example results"):
        rows = []
        for r in per_example:
            row = {}
            if "text" in r:
                row["Text"] = r["text"][:55] + "..."
            elif "question" in r:
                row["Question"] = r["question"][:55] + "..."
            row["Expected"] = str(r.get("expected", "")).upper()
            for m in models:
                lbl = r.get(f"{m}_label", r.get(f"{m}_extracted", "?"))
                ok = "Y" if r.get(f"{m}_correct") else "N"
                short = labels.get(m, m).split("(")[0].strip()
                row[short] = f"{lbl} [{ok}]"
            row["Category"] = r.get("category", "").replace("_", " ").title()
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Token Usage & Cost
    tok_data = {m: summary.get(m, {}).get("total_tokens", 0) for m in models}
    if any(v for v in tok_data.values()):
        st.markdown("---")
        st.markdown("##### Token Usage & Cost")
        tok_cols = st.columns(len(models))
        for col, m in zip(tok_cols, models):
            s = summary.get(m, {})
            with col:
                avg_tok = s.get("avg_tokens_per_query", 0)
                cost_1k = s.get("cost_per_1k_queries_usd", 0)
                st.metric(labels.get(m, m).split("(")[0].strip(),
                          f"{avg_tok:,} tok/query")
                st.caption(f"Cost/1K queries: ${cost_1k:.4f}")

    # F1 score (classification tasks)
    f1_data = {m: summary.get(m, {}).get("f1_macro") for m in models}
    if any(v is not None for v in f1_data.values()):
        st.markdown("##### Quality Metrics: F1 Score")
        f1_cols = st.columns(len(models))
        for col, m in zip(f1_cols, models):
            s = summary.get(m, {})
            with col:
                f1 = s.get("f1_macro", 0)
                prec = s.get("precision_macro", 0)
                rec = s.get("recall_macro", 0)
                st.metric(labels.get(m, m).split("(")[0].strip(),
                          f"F1: {f1:.3f}")
                st.caption(f"P: {prec:.3f} / R: {rec:.3f}")

    # MAPE (numerical tasks)
    mape_data = {m: summary.get(m, {}).get("mape") for m in models}
    if any(v is not None for v in mape_data.values()):
        st.markdown("##### Quality Metrics: MAPE")
        mape_cols = st.columns(len(models))
        for col, m in zip(mape_cols, models):
            s = summary.get(m, {})
            with col:
                mape = s.get("mape")
                if mape is not None:
                    st.metric(labels.get(m, m).split("(")[0].strip(),
                              f"{mape:.1f}%")

    # =====================================================================
    # LLM-as-Judge Assessment
    # =====================================================================
    if judge_summary:
        st.markdown("---")
        st.markdown("##### LLM-as-Judge Assessment")
        st.caption("Structured scoring: Correctness (1-5), Reasoning Quality (1-5), Faithfulness (1-5)")

        # Summary scores
        judge_cols = st.columns(len(models))
        for col, m in zip(judge_cols, models):
            js = judge_summary.get(m, {})
            with col:
                overall = js.get("overall", 0)
                count = js.get("count", 0)
                short = labels.get(m, m).split("(")[0].strip()
                st.metric(short, f"{overall:.2f} / 5.00",
                          delta=f"({count} judged)")
                st.caption(
                    f"C: {js.get('correctness', 0):.1f} | "
                    f"R: {js.get('reasoning_quality', 0):.1f} | "
                    f"F: {js.get('faithfulness', 0):.1f}"
                )

        # Judge radar chart
        import plotly.graph_objects as go
        dimensions = ["Correctness", "Reasoning Quality", "Faithfulness"]
        fig_radar = go.Figure()
        for m in models:
            js = judge_summary.get(m, {})
            if js.get("count", 0) > 0:
                vals = [
                    js.get("correctness", 0),
                    js.get("reasoning_quality", 0),
                    js.get("faithfulness", 0),
                ]
                vals.append(vals[0])  # close the polygon
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=dimensions + [dimensions[0]],
                    name=labels.get(m, m).split("(")[0].strip(),
                    line_color=COLORS.get(m, "#999"),
                    fill="toself", opacity=0.3,
                ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            title="Judge Scores Radar",
        )
        st.plotly_chart(fig_radar, use_container_width=True, key=f"adv_radar_{title}")

        # Per-example judge details
        with st.expander("Per-example judge scores"):
            jrows = []
            for r in per_example:
                jrow = {"ID": r.get("id", "")}
                if "text" in r:
                    jrow["Text"] = r["text"][:40] + "..."
                elif "question" in r:
                    jrow["Question"] = r["question"][:40] + "..."
                for m in models:
                    judge = r.get(f"{m}_judge")
                    short = labels.get(m, m).split("(")[0].strip()
                    if judge and isinstance(judge, dict):
                        jrow[f"{short} C"] = judge.get("correctness", "")
                        jrow[f"{short} R"] = judge.get("reasoning_quality", "")
                        jrow[f"{short} F"] = judge.get("faithfulness", "")
                    else:
                        jrow[f"{short} C"] = "-"
                        jrow[f"{short} R"] = "-"
                        jrow[f"{short} F"] = "-"
                jrows.append(jrow)
            if jrows:
                st.dataframe(pd.DataFrame(jrows), use_container_width=True)
            else:
                st.info("No judge scores available. Run with --with-judge to enable.")


# =========================================================================
# Live benchmark runners (Streamlit wrappers)
# =========================================================================
def run_live_adversarial_sentiment(with_judge=False):
    """Run adversarial sentiment benchmark with live progress."""
    from adversarial_benchmark import (
        get_adversarial_sentiment_cases,
        run_single_adversarial_sentiment_case,
        SENTIMENT_MODEL_NAMES,
    )
    from benchmark import compute_live_stats

    cases = get_adversarial_sentiment_cases()
    total = len(cases)
    results = []

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name
        judge_model = get_judge_model_name()
        if not judge_model:
            st.warning("No judge model available via Ollama")
            with_judge = False
        else:
            st.info(f"LLM Judge: **{judge_model}**")

    progress = st.progress(0, text="Starting adversarial sentiment benchmark...")
    metrics_ph = st.empty()
    table_ph = st.empty()

    for i, case in enumerate(cases):
        progress.progress((i + 1) / total,
                          text=f"Case {i+1}/{total}: {case['text'][:50]}... [{case['category']}]")
        row = run_single_adversarial_sentiment_case(case, with_judge, judge_model)
        results.append(row)

        stats = compute_live_stats(results, SENTIMENT_MODEL_NAMES)
        with metrics_ph.container():
            mcols = st.columns(4)
            for col, m in zip(mcols, SENTIMENT_MODEL_NAMES):
                s = stats[m]
                col.metric(m.replace("_", " ").title(),
                           f"{s['accuracy']}%",
                           delta=f"{s['correct']}/{s['total']}")

        _update_live_adversarial_table(table_ph, results, SENTIMENT_MODEL_NAMES, "text")

    progress.progress(1.0, text="Adversarial sentiment benchmark complete!")
    return results


def run_live_adversarial_numerical(with_judge=False):
    """Run adversarial numerical benchmark with live progress."""
    from adversarial_benchmark import (
        get_adversarial_numerical_cases,
        run_single_adversarial_numerical_case,
        NUMERICAL_MODEL_NAMES,
    )
    from benchmark import compute_live_stats
    from demo_utils import has_llm

    if not has_llm():
        st.error("Ollama not available. Start Ollama to run this benchmark.")
        return []

    cases = get_adversarial_numerical_cases()
    total = len(cases)
    results = []

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name
        judge_model = get_judge_model_name()
        if not judge_model:
            st.warning("No judge model available via Ollama")
            with_judge = False
        else:
            st.info(f"LLM Judge: **{judge_model}**")

    progress = st.progress(0, text="Starting adversarial numerical benchmark...")
    metrics_ph = st.empty()
    table_ph = st.empty()

    for i, case in enumerate(cases):
        progress.progress((i + 1) / total,
                          text=f"Case {i+1}/{total}: {case['question'][:50]}... [{case['category']}]")
        row = run_single_adversarial_numerical_case(case, with_judge, judge_model)
        results.append(row)

        stats = compute_live_stats(results, NUMERICAL_MODEL_NAMES)
        with metrics_ph.container():
            mcols = st.columns(4)
            for col, m in zip(mcols, NUMERICAL_MODEL_NAMES):
                s = stats[m]
                col.metric(m.replace("_", " ").title(),
                           f"{s['accuracy']}%",
                           delta=f"{s['correct']}/{s['total']}")

        _update_live_adversarial_table(table_ph, results, NUMERICAL_MODEL_NAMES, "question")

    progress.progress(1.0, text="Adversarial numerical benchmark complete!")
    return results


def run_live_adversarial_financial_ratios(with_judge=False):
    """Run adversarial financial ratios benchmark with live progress."""
    from adversarial_benchmark import (
        get_adversarial_financial_ratio_cases,
        run_single_adversarial_numerical_case,
        FINANCIAL_RATIO_MODEL_NAMES,
    )
    from benchmark import compute_live_stats
    from demo_utils import has_llm

    if not has_llm():
        st.error("Ollama not available. Start Ollama to run this benchmark.")
        return []

    cases = get_adversarial_financial_ratio_cases()
    total = len(cases)
    results = []

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name
        judge_model = get_judge_model_name()
        if not judge_model:
            st.warning("No judge model available via Ollama")
            with_judge = False
        else:
            st.info(f"LLM Judge: **{judge_model}**")

    progress = st.progress(0, text="Starting adversarial financial ratios benchmark...")
    metrics_ph = st.empty()
    table_ph = st.empty()

    for i, case in enumerate(cases):
        progress.progress((i + 1) / total,
                          text=f"Case {i+1}/{total}: {case['question'][:50]}... [{case['category']}]")
        row = run_single_adversarial_numerical_case(case, with_judge, judge_model)
        results.append(row)

        stats = compute_live_stats(results, FINANCIAL_RATIO_MODEL_NAMES)
        with metrics_ph.container():
            mcols = st.columns(4)
            for col, m in zip(mcols, FINANCIAL_RATIO_MODEL_NAMES):
                s = stats[m]
                col.metric(m.replace("_", " ").title(),
                           f"{s['accuracy']}%",
                           delta=f"{s['correct']}/{s['total']}")

        _update_live_adversarial_table(table_ph, results, FINANCIAL_RATIO_MODEL_NAMES, "question")

    progress.progress(1.0, text="Adversarial financial ratios benchmark complete!")
    return results


def run_live_adversarial_spam(with_judge=False):
    """Run adversarial spam benchmark with live progress."""
    from adversarial_benchmark import (
        get_adversarial_spam_cases,
        run_single_adversarial_spam_case,
        SPAM_MODEL_NAMES,
    )
    from benchmark import compute_live_stats

    cases = get_adversarial_spam_cases()
    total = len(cases)
    results = []

    judge_model = None
    if with_judge:
        from llm_judge import get_judge_model_name
        judge_model = get_judge_model_name()
        if not judge_model:
            st.warning("No judge model available via Ollama")
            with_judge = False
        else:
            st.info(f"LLM Judge: **{judge_model}**")

    progress = st.progress(0, text="Starting adversarial spam benchmark...")
    metrics_ph = st.empty()
    table_ph = st.empty()

    for i, case in enumerate(cases):
        progress.progress((i + 1) / total,
                          text=f"Case {i+1}/{total}: {case['text'][:50]}... [{case['category']}]")
        row = run_single_adversarial_spam_case(case, with_judge, judge_model)
        results.append(row)

        stats = compute_live_stats(results, SPAM_MODEL_NAMES)
        with metrics_ph.container():
            mcols = st.columns(4)
            for col, m in zip(mcols, SPAM_MODEL_NAMES):
                s = stats[m]
                col.metric(m.replace("_", " ").title(),
                           f"{s['accuracy']}%",
                           delta=f"{s['correct']}/{s['total']}")

        _update_live_adversarial_table(table_ph, results, SPAM_MODEL_NAMES, "text")

    progress.progress(1.0, text="Adversarial spam benchmark complete!")
    return results


def _update_live_adversarial_table(placeholder, results, models, text_key):
    """Update the live results table."""
    table_rows = []
    for r in results:
        trow = {}
        if text_key in r:
            trow[text_key.title()] = r[text_key][:45] + "..."
        trow["Expected"] = str(r.get("expected", "")).upper()
        trow["Category"] = r.get("category", "").replace("_", " ").title()
        for m in models:
            if f"{m}_label" in r:
                ok = "Y" if r.get(f"{m}_correct") else "N"
                lbl = r.get(f"{m}_label", "?")
                trow[m.split("(")[0].strip()] = f"{lbl} [{ok}]"
            elif f"{m}_extracted" in r:
                ok = "Y" if r.get(f"{m}_correct") else "N"
                ext = r.get(f"{m}_extracted", "?")
                trow[m.split("(")[0].strip()] = f"{ext} [{ok}]"
            else:
                trow[m.split("(")[0].strip()] = "..."
        table_rows.append(trow)
    placeholder.dataframe(pd.DataFrame(table_rows), use_container_width=True)


def _save_adversarial_results(section_key, section_data, with_judge=False,
                               judge_summary=None):
    """Save results for one section, preserving others."""
    existing = {}
    if ADVERSARIAL_RESULTS_PATH.exists():
        try:
            with open(ADVERSARIAL_RESULTS_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing_sections = existing.get("sections", {})
    existing_judge = existing.get("judge_summaries", {})

    existing_sections[section_key] = section_data

    if judge_summary:
        existing_judge[section_key] = judge_summary

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_type": "adversarial_stress_test",
        "with_judge": with_judge or existing.get("with_judge", False),
        "judge_summaries": existing_judge,
        "sections": existing_sections,
    }

    ADVERSARIAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ADVERSARIAL_RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)


# =========================================================================
# Main UI
# =========================================================================

# Judge toggle
with_judge = st.sidebar.checkbox(
    "Enable LLM-as-Judge",
    value=False,
    help="Use OpenAI GPT-4o to evaluate responses with structured scoring (Correctness, Reasoning Quality, Faithfulness). Requires OPENAI_API_KEY.",
)

if with_judge:
    from llm_judge import get_judge_model_name
    jm = get_judge_model_name()
    if jm:
        st.sidebar.success(f"Judge model: **{jm}**")
    else:
        st.sidebar.warning("No judge model found via Ollama")

# Labels
from adversarial_benchmark import (
    ADVERSARIAL_SENTIMENT_LABELS,
    ADVERSARIAL_NUMERICAL_LABELS,
    ADVERSARIAL_SPAM_LABELS,
)

SENTIMENT_LABELS = ADVERSARIAL_SENTIMENT_LABELS
NUMERICAL_LABELS = ADVERSARIAL_NUMERICAL_LABELS
FINANCIAL_RATIO_LABELS = ADVERSARIAL_NUMERICAL_LABELS
SPAM_LABELS = ADVERSARIAL_SPAM_LABELS

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sentiment (BERT 110M)",
    "Numerical (Llama2 7B)",
    "Financial Ratios (Llama2 7B)",
    "Spam (DistilBERT 66M)",
    "Cross-Experiment Analysis",
])


# ---- Tab 1: Adversarial Sentiment ----
with tab1:
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_live_s = st.button("Run Adversarial Benchmark", type="primary",
                               use_container_width=True, key="adv_live_sent")
    with col_btn2:
        show_saved_s = st.button("Show Saved Results", use_container_width=True,
                                  key="adv_saved_sent")

    if run_live_s:
        sent_results = run_live_adversarial_sentiment(with_judge)

        from benchmark import compute_section_summary
        sent_summary = compute_section_summary(sent_results, ["base", "finbert", "rag", "hybrid"])

        judge_sum = None
        if with_judge:
            from llm_judge import compute_judge_summary
            judge_sum = compute_judge_summary(sent_results, ["base", "finbert", "rag", "hybrid"])

        section_data = {
            "title": "Adversarial Sentiment Classification",
            "architecture": "BERT-base-uncased (110M parameters)",
            "models": ["base", "finbert", "rag", "hybrid"],
            "model_labels": SENTIMENT_LABELS,
            "summary": sent_summary,
            "results": sent_results,
        }
        _save_adversarial_results("adversarial_sentiment", section_data,
                                   with_judge, judge_sum)
        st.success("Adversarial sentiment results saved!")

    elif show_saved_s or (not run_live_s):
        if adv_results_data and "adversarial_sentiment" in adv_results_data.get("sections", {}):
            sections = adv_results_data["sections"]
            judge_sums = adv_results_data.get("judge_summaries", {})
            st.caption(f"Saved results from: {adv_results_data.get('timestamp', 'unknown')}")
            render_adversarial_section(
                sections["adversarial_sentiment"],
                judge_sums.get("adversarial_sentiment"),
            )
        else:
            st.info("No saved adversarial sentiment results. Click **Run Adversarial Benchmark** to generate them.")


# ---- Tab 2: Adversarial Numerical ----
with tab2:
    col_btn3, col_btn4 = st.columns(2)
    with col_btn3:
        run_live_n = st.button("Run Adversarial Benchmark", type="primary",
                               use_container_width=True, key="adv_live_num")
    with col_btn4:
        show_saved_n = st.button("Show Saved Results", use_container_width=True,
                                  key="adv_saved_num")

    if run_live_n:
        num_results = run_live_adversarial_numerical(with_judge)

        if num_results:
            from benchmark import compute_section_summary
            num_summary = compute_section_summary(num_results, ["base", "finetuned", "rag", "hybrid"])

            judge_sum = None
            if with_judge:
                from llm_judge import compute_judge_summary
                judge_sum = compute_judge_summary(num_results, ["base", "finetuned", "rag", "hybrid"])

            section_data = {
                "title": "Adversarial Numerical Reasoning",
                "architecture": "Llama2-7B (7B parameters)",
                "models": ["base", "finetuned", "rag", "hybrid"],
                "model_labels": NUMERICAL_LABELS,
                "summary": num_summary,
                "results": num_results,
            }
            _save_adversarial_results("adversarial_numerical", section_data,
                                       with_judge, judge_sum)
            st.success("Adversarial numerical results saved!")

    elif show_saved_n or (not run_live_n):
        if adv_results_data and "adversarial_numerical" in adv_results_data.get("sections", {}):
            sections = adv_results_data["sections"]
            judge_sums = adv_results_data.get("judge_summaries", {})
            st.caption(f"Saved results from: {adv_results_data.get('timestamp', 'unknown')}")
            render_adversarial_section(
                sections["adversarial_numerical"],
                judge_sums.get("adversarial_numerical"),
            )
        else:
            st.info("No saved adversarial numerical results. Click **Run Adversarial Benchmark** to generate them.")


# ---- Tab 3: Adversarial Financial Ratios ----
with tab3:
    col_btn5, col_btn6 = st.columns(2)
    with col_btn5:
        run_live_fr = st.button("Run Adversarial Benchmark", type="primary",
                                use_container_width=True, key="adv_live_fr")
    with col_btn6:
        show_saved_fr = st.button("Show Saved Results", use_container_width=True,
                                   key="adv_saved_fr")

    if run_live_fr:
        fr_results = run_live_adversarial_financial_ratios(with_judge)

        if fr_results:
            from benchmark import compute_section_summary
            fr_summary = compute_section_summary(fr_results, ["base", "finetuned", "rag", "hybrid"])

            judge_sum = None
            if with_judge:
                from llm_judge import compute_judge_summary
                judge_sum = compute_judge_summary(fr_results, ["base", "finetuned", "rag", "hybrid"])

            section_data = {
                "title": "Adversarial Financial Ratios",
                "architecture": "Llama2-7B (7B parameters)",
                "models": ["base", "finetuned", "rag", "hybrid"],
                "model_labels": FINANCIAL_RATIO_LABELS,
                "summary": fr_summary,
                "results": fr_results,
            }
            _save_adversarial_results("adversarial_financial_ratios", section_data,
                                       with_judge, judge_sum)
            st.success("Adversarial financial ratios results saved!")

    elif show_saved_fr or (not run_live_fr):
        if adv_results_data and "adversarial_financial_ratios" in adv_results_data.get("sections", {}):
            sections = adv_results_data["sections"]
            judge_sums = adv_results_data.get("judge_summaries", {})
            st.caption(f"Saved results from: {adv_results_data.get('timestamp', 'unknown')}")
            render_adversarial_section(
                sections["adversarial_financial_ratios"],
                judge_sums.get("adversarial_financial_ratios"),
            )
        else:
            st.info("No saved adversarial financial ratio results. Click **Run Adversarial Benchmark** to generate them.")


# ---- Tab 4: Adversarial Spam ----
with tab4:
    col_btn7, col_btn8 = st.columns(2)
    with col_btn7:
        run_live_sp = st.button("Run Adversarial Benchmark", type="primary",
                                use_container_width=True, key="adv_live_spam")
    with col_btn8:
        show_saved_sp = st.button("Show Saved Results", use_container_width=True,
                                   key="adv_saved_spam")

    if run_live_sp:
        spam_results = run_live_adversarial_spam(with_judge)

        from benchmark import compute_section_summary
        spam_summary = compute_section_summary(spam_results, ["base", "finetuned", "rag", "hybrid"])

        judge_sum = None
        if with_judge:
            from llm_judge import compute_judge_summary
            judge_sum = compute_judge_summary(spam_results, ["base", "finetuned", "rag", "hybrid"])

        section_data = {
            "title": "Adversarial Spam Detection",
            "architecture": "DistilBERT-base-uncased (66M parameters)",
            "models": ["base", "finetuned", "rag", "hybrid"],
            "model_labels": SPAM_LABELS,
            "summary": spam_summary,
            "results": spam_results,
        }
        _save_adversarial_results("adversarial_spam", section_data,
                                   with_judge, judge_sum)
        st.success("Adversarial spam results saved!")

    elif show_saved_sp or (not run_live_sp):
        if adv_results_data and "adversarial_spam" in adv_results_data.get("sections", {}):
            sections = adv_results_data["sections"]
            judge_sums = adv_results_data.get("judge_summaries", {})
            st.caption(f"Saved results from: {adv_results_data.get('timestamp', 'unknown')}")
            render_adversarial_section(
                sections["adversarial_spam"],
                judge_sums.get("adversarial_spam"),
            )
        else:
            st.info("No saved adversarial spam results. Click **Run Adversarial Benchmark** to generate them.")


# ---- Tab 5: Cross-Experiment Analysis ----
with tab5:
    st.subheader("Cross-Experiment Adversarial Analysis")

    if adv_results_data and "sections" in adv_results_data:
        sections = adv_results_data["sections"]
        judge_sums = adv_results_data.get("judge_summaries", {})

        # Summary table across all experiments
        summary_rows = []
        section_configs = [
            ("adversarial_sentiment", "Sentiment (BERT 110M)", ["base", "finbert", "rag", "hybrid"]),
            ("adversarial_numerical", "Numerical (Llama2 7B)", ["base", "finetuned", "rag", "hybrid"]),
            ("adversarial_financial_ratios", "Financial Ratios (Llama2 7B)", ["base", "finetuned", "rag", "hybrid"]),
            ("adversarial_spam", "Spam (DistilBERT 66M)", ["base", "finetuned", "rag", "hybrid"]),
        ]

        for sec_key, sec_name, sec_models in section_configs:
            sec = sections.get(sec_key)
            if not sec:
                continue
            s = sec.get("summary", {})
            row = {"Experiment": sec_name}
            best_acc = 0
            best_model = ""
            for m in sec_models:
                ms = s.get(m, {})
                acc = ms.get("accuracy", 0)
                label = sec["model_labels"].get(m, m).split("(")[0].strip()
                row[label] = f"{acc}%"
                if acc > best_acc:
                    best_acc = acc
                    best_model = label
            row["Winner"] = best_model
            summary_rows.append(row)

        if summary_rows:
            st.markdown("##### Accuracy Summary (All Adversarial Experiments)")
            st.table(pd.DataFrame(summary_rows))

        # Category breakdown across experiments
        st.markdown("##### Accuracy by Adversarial Category (All Experiments)")
        cat_summary_rows = []
        for sec_key, sec_name, sec_models in section_configs:
            sec = sections.get(sec_key)
            if not sec:
                continue
            s = sec.get("summary", {})
            for key, val in sorted(s.items()):
                if not key.startswith("category_"):
                    continue
                cat = key.replace("category_", "").replace("_", " ").title()
                row = {"Experiment": sec_name, "Category": cat, "Cases": val["total"]}
                for m in sec_models:
                    label = sec["model_labels"].get(m, m).split("(")[0].strip()
                    row[label] = f"{val.get(f'{m}_accuracy', 0)}%"
                cat_summary_rows.append(row)

        if cat_summary_rows:
            st.dataframe(pd.DataFrame(cat_summary_rows), use_container_width=True)

        # Judge comparison across experiments
        if judge_sums:
            st.markdown("##### LLM Judge Scores (All Experiments)")
            judge_rows = []
            for sec_key, sec_name, sec_models in section_configs:
                js = judge_sums.get(sec_key)
                if not js:
                    continue
                sec = sections.get(sec_key, {})
                for m in sec_models:
                    jm = js.get(m, {})
                    if jm.get("count", 0) > 0:
                        label = sec.get("model_labels", {}).get(m, m).split("(")[0].strip()
                        judge_rows.append({
                            "Experiment": sec_name,
                            "Model": label,
                            "Correctness": f"{jm['correctness']:.1f}",
                            "Reasoning": f"{jm['reasoning_quality']:.1f}",
                            "Faithfulness": f"{jm['faithfulness']:.1f}",
                            "Overall": f"{jm['overall']:.2f}",
                        })
            if judge_rows:
                st.dataframe(pd.DataFrame(judge_rows), use_container_width=True)

        # Adversarial robustness comparison: normal vs adversarial
        st.markdown("---")
        st.markdown("##### Normal vs Adversarial Accuracy Drop")
        st.caption("Compare standard benchmark accuracy to adversarial stress test accuracy")

        normal_results_path = Path(__file__).parent.parent.parent / "data" / "benchmark_results.json"
        if normal_results_path.exists():
            with open(normal_results_path) as f:
                normal_data = json.load(f)

            normal_sections = normal_data.get("sections", {})
            drop_rows = []

            comparisons = [
                ("bert_110m_sentiment", "adversarial_sentiment", "Sentiment (BERT 110M)", ["base", "finbert", "rag", "hybrid"]),
                ("llama2_7b_numerical", "adversarial_numerical", "Numerical (Llama2 7B)", ["base", "finetuned", "rag", "hybrid"]),
                ("llama2_7b_financial_ratios", "adversarial_financial_ratios", "Financial Ratios (Llama2 7B)", ["base", "finetuned", "rag", "hybrid"]),
                ("distilbert_66m_spam", "adversarial_spam", "Spam (DistilBERT 66M)", ["base", "finetuned", "rag", "hybrid"]),
            ]

            for norm_key, adv_key, exp_name, exp_models in comparisons:
                norm_sec = normal_sections.get(norm_key, {})
                adv_sec = sections.get(adv_key, {})
                if not norm_sec or not adv_sec:
                    continue

                norm_summary = norm_sec.get("summary", {})
                adv_summary = adv_sec.get("summary", {})

                for m in exp_models:
                    norm_acc = norm_summary.get(m, {}).get("accuracy", 0)
                    adv_acc = adv_summary.get(m, {}).get("accuracy", 0)
                    drop = norm_acc - adv_acc

                    label = adv_sec.get("model_labels", {}).get(m, m).split("(")[0].strip()
                    drop_rows.append({
                        "Experiment": exp_name,
                        "Model": label,
                        "Normal Accuracy": f"{norm_acc}%",
                        "Adversarial Accuracy": f"{adv_acc}%",
                        "Drop": f"{drop:+.1f}pp",
                    })

            if drop_rows:
                st.dataframe(pd.DataFrame(drop_rows), use_container_width=True)
            else:
                st.info("Run both normal and adversarial benchmarks to see the comparison.")
        else:
            st.info("Run the standard benchmark first to enable normal vs adversarial comparison.")
    else:
        st.info("No adversarial results saved yet. Run experiments in the other tabs first.")


st.divider()
st.markdown(
    "**Adversarial Stress Test**: 30 cases per experiment designed to expose model weaknesses. "
    "Categories: noisy retrieval, knowledge conflict, out-of-distribution. "
    "Optional LLM-as-Judge provides structured quality assessment beyond simple accuracy."
)
