"""
Benchmark Results Page
Four controlled experiments, each comparing Base vs Fine-Tuned vs RAG vs Hybrid
on the SAME architecture.

Section 1: BERT 110M (Sentiment) -- Base BERT vs FinBERT vs BERT+RAG vs Hybrid
Section 2: Llama2 7B (Numerical)  -- Base Llama2 vs Expert prompt vs Llama2+RAG vs Hybrid
Section 3: Llama2 7B (Financial Ratios) -- Base Llama2 vs FinQA-7B vs Llama2+RAG vs Hybrid
Section 4: DistilBERT 66M (Spam Detection) -- Base DistilBERT vs Fine-tuned vs RAG vs Hybrid

Supports both pre-loaded results and live real-time benchmark execution.
"""
import streamlit as st
import pandas as pd
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Benchmark Results", page_icon="FT", layout="wide")

st.title("Benchmark Results (Our Own Measurements)")
st.markdown("""
Four controlled experiments comparing **Base Model vs Fine-Tuned vs RAG vs Hybrid**.
Each experiment uses the **same architecture and parameter count** so the
only variable is the approach.

| Experiment | Architecture | Approaches | Task |
|-----------|-------------|-----------|------|
| **Section 1** | BERT-base (110M params) | Base, FinBERT, RAG, Hybrid | Sentiment classification |
| **Section 2** | Llama2-7B (7B params) | Base, FinQA-7B, RAG (base), Hybrid (FinQA-7B+RAG) | Numerical reasoning |
| **Section 3** | Llama2-7B (7B params) | Base, FinQA-7B, RAG (base), Hybrid (FinQA-7B+RAG) | Financial ratio calculation |
| **Section 4** | DistilBERT (66M params) | Base, Fine-tuned, RAG, Hybrid | Spam detection |

Every number was measured by running our actual models. Nothing from papers.
""")

RESULTS_PATH = Path(__file__).parent.parent.parent / "data" / "benchmark_results.json"
TEST_CASES_PATH = Path(__file__).parent.parent.parent / "data" / "benchmark_test_cases.json"

COLORS = {"finbert": "#28a745", "finetuned": "#28a745",
          "base": "#dc3545", "rag": "#007bff", "hybrid": "#ff8c00"}

SENTIMENT_LABELS = {
    "base": "Base BERT (no FT, no RAG)",
    "finbert": "FinBERT (fine-tuned)",
    "rag": "BERT + RAG (retrieval + voting)",
    "hybrid": "FinBERT + RAG (hybrid)",
}

NUMERICAL_LABELS = {
    "base": "Base Llama2-7B",
    "finetuned": "FinQA-7B (fine-tuned Llama2-7B)",
    "rag": "Llama2-7B + RAG (base model)",
    "hybrid": "FinQA-7B + RAG (fine-tuned + retrieval)",
}

FINANCIAL_RATIO_LABELS = {
    "base": "Base Llama2-7B",
    "finetuned": "FinQA-7B (fine-tuned Llama2-7B)",
    "rag": "Llama2-7B + RAG (base model)",
    "hybrid": "FinQA-7B + RAG (fine-tuned + retrieval)",
}

SPAM_LABELS = {
    "base": "Base DistilBERT (no FT, no RAG)",
    "finetuned": "Fine-tuned DistilBERT (spam-trained)",
    "rag": "DistilBERT + RAG (retrieval + voting)",
    "hybrid": "Fine-tuned DistilBERT + RAG (hybrid)",
}

# Load pre-existing results
results_data = None
if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        results_data = json.load(f)

if results_data and "sections" not in results_data:
    results_data = None


# =========================================================================
# Render a benchmark section (static, from saved data)
# =========================================================================
def render_section(section_data):
    import plotly.graph_objects as go

    title = section_data["title"]
    arch = section_data["architecture"]
    models = section_data["models"]
    labels = section_data["model_labels"]
    summary = section_data["summary"]
    per_example = section_data["results"]

    st.subheader(f"{title}")
    st.caption(f"Architecture: {arch} -- same for all four approaches")

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
    fig.update_layout(title=f"Accuracy -- {arch}", barmode="group",
                       yaxis_range=[0, 105])
    st.plotly_chart(fig, use_container_width=True, key=f"acc_chart_{title}")

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
        st.plotly_chart(fig2, use_container_width=True, key=f"lat_chart_{title}")

    # Confidence chart (sentiment only)
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
        fig_conf.update_layout(title="Average Confidence", barmode="group",
                                yaxis_range=[0, 1.05])
        st.plotly_chart(fig_conf, use_container_width=True, key=f"conf_chart_{title}")

    # Per-example table
    with st.expander("Per-example results"):
        rows = []
        for r in per_example:
            row = {}
            if "text" in r:
                row["Text"] = r["text"][:60] + "..."
            elif "question" in r:
                row["Question"] = r["question"][:60] + "..."
            row["Expected"] = str(r.get("expected", "")).upper()
            for m in models:
                lbl = r.get(f"{m}_label", r.get(f"{m}_extracted", "?"))
                ok = "Y" if r.get(f"{m}_correct") else "N"
                short = labels.get(m, m).split("(")[0].strip()
                row[short] = f"{lbl} [{ok}]"
            row["Category"] = r.get("category", "")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Category breakdown
    cat_rows = []
    for key, val in sorted(summary.items()):
        if not key.startswith("category_"):
            continue
        cat = key.replace("category_", "").replace("_", " ").title()
        row = {"Category": cat, "Cases": val["total"]}
        for m in models:
            row[labels.get(m, m).split("(")[0].strip()] = f"{val.get(f'{m}_accuracy', 0)}%"
        cat_rows.append(row)

    if cat_rows:
        with st.expander("Category breakdown"):
            st.table(pd.DataFrame(cat_rows))

    # Token Usage & Cost metrics
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
                tps = s.get("avg_throughput_tps", 0)
                st.metric(labels.get(m, m).split("(")[0].strip(),
                          f"{avg_tok:,} tok/query")
                st.caption(f"Cost/1K queries: ${cost_1k:.4f}")
                if tps:
                    st.caption(f"Throughput: {tps:,.0f} tok/s")

        # Token chart
        fig_tok = go.Figure()
        for m in models:
            s = summary.get(m, {})
            fig_tok.add_trace(go.Bar(
                name=labels.get(m, m).split("(")[0].strip(),
                x=["Avg Tokens/Query"],
                y=[s.get("avg_tokens_per_query", 0)],
                marker_color=COLORS.get(m, "#999"),
                text=[f"{s.get('avg_tokens_per_query', 0):,}"],
                textposition="auto",
            ))
        fig_tok.update_layout(title="Average Tokens per Query", barmode="group")
        st.plotly_chart(fig_tok, use_container_width=True, key=f"tok_chart_{title}")

    # F1 score (sentiment)
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

    # MAPE (numerical)
    mape_data = {m: summary.get(m, {}).get("mape") for m in models}
    if any(v is not None for v in mape_data.values()):
        st.markdown("##### Quality Metrics: Mean Absolute % Error")
        mape_cols = st.columns(len(models))
        for col, m in zip(mape_cols, models):
            s = summary.get(m, {})
            with col:
                mape = s.get("mape", 0)
                st.metric(labels.get(m, m).split("(")[0].strip(),
                          f"MAPE: {mape:.1f}%",
                          delta=None)

    if cat_rows:
        cats = [r["Category"] for r in cat_rows]
        fig3 = go.Figure()
        for m in models:
            accs = [float(r[labels.get(m, m).split("(")[0].strip()].replace("%", ""))
                    for r in cat_rows]
            fig3.add_trace(go.Bar(
                name=labels.get(m, m), x=cats, y=accs,
                marker_color=COLORS.get(m, "#999"),
            ))
        fig3.update_layout(title="Accuracy by Category", barmode="group",
                           yaxis_range=[0, 105])
        st.plotly_chart(fig3, use_container_width=True, key=f"cat_chart_{title}")


# =========================================================================
# Live benchmark runner -- real-time case-by-case execution
# =========================================================================
def run_live_sentiment_benchmark():
    """Run BERT 110M sentiment benchmark with live per-case UI updates."""
    import plotly.graph_objects as go
    from benchmark import (
        get_sentiment_cases, run_single_sentiment_case,
        compute_live_stats, compute_section_summary,
        SENTIMENT_MODEL_NAMES,
    )

    cases = get_sentiment_cases()
    models = SENTIMENT_MODEL_NAMES
    labels = SENTIMENT_LABELS

    st.subheader("BERT 110M -- Sentiment Classification (LIVE)")
    st.caption("Architecture: BERT-base-uncased (110M parameters) -- same for all four approaches")

    # Placeholders for live-updating widgets
    progress = st.progress(0, text="Starting benchmark...")
    metric_cols = st.columns(len(models))
    metric_placeholders = {}
    for col, m in zip(metric_cols, models):
        with col:
            metric_placeholders[m] = st.empty()
            metric_placeholders[m].metric(labels[m], "0%", delta="0/0")

    chart_placeholder = st.empty()
    latency_chart_placeholder = st.empty()
    table_placeholder = st.empty()

    all_results = []
    start_time = time.perf_counter()

    for i, case in enumerate(cases):
        progress.progress(
            (i) / len(cases),
            text=f"Running case {i+1}/{len(cases)}: {case['text'][:50]}..."
        )

        row = run_single_sentiment_case(case)
        all_results.append(row)

        # Update live stats
        stats = compute_live_stats(all_results, models)

        # Update metrics
        for m in models:
            s = stats[m]
            metric_placeholders[m].metric(
                labels[m],
                f"{s['accuracy']}%",
                delta=f"{s['correct']}/{s['total']}",
            )

        # Update accuracy chart
        fig = go.Figure()
        for m in models:
            s = stats[m]
            fig.add_trace(go.Bar(
                name=labels[m], x=["Accuracy (%)"],
                y=[s["accuracy"]],
                marker_color=COLORS.get(m, "#999"),
                text=[f"{s['accuracy']}%"], textposition="auto",
            ))
        fig.update_layout(
            title=f"Accuracy after {len(all_results)}/{len(cases)} cases",
            barmode="group", yaxis_range=[0, 105],
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Update latency chart
        fig2 = go.Figure()
        for m in models:
            s = stats[m]
            if s["avg_latency_ms"]:
                fig2.add_trace(go.Bar(
                    name=labels[m], x=["Avg Latency (ms)"],
                    y=[s["avg_latency_ms"]],
                    marker_color=COLORS.get(m, "#999"),
                    text=[f"{s['avg_latency_ms']:.0f}ms"], textposition="auto",
                ))
        fig2.update_layout(title="Average Latency (running)", barmode="group")
        latency_chart_placeholder.plotly_chart(fig2, use_container_width=True)

        # Update results table
        table_rows = []
        for r in all_results:
            trow = {
                "Text": r["text"][:55] + "...",
                "Expected": r["expected"].upper(),
            }
            for m in models:
                lbl = r.get(f"{m}_label", "?")
                ok = "Y" if r.get(f"{m}_correct") else "N"
                lat = r.get(f"{m}_latency_ms", 0)
                short = labels[m].split("(")[0].strip()
                trow[short] = f"{lbl} [{ok}] {lat:.0f}ms"
            trow["Category"] = r.get("category", "")
            table_rows.append(trow)
        table_placeholder.dataframe(
            pd.DataFrame(table_rows), use_container_width=True
        )

    elapsed = (time.perf_counter() - start_time) * 1000
    progress.progress(1.0, text=f"Benchmark complete! ({elapsed:.0f}ms total)")

    return all_results


def run_live_numerical_benchmark():
    """Run Llama2 7B numerical benchmark with per-MODEL live UI updates."""
    import plotly.graph_objects as go
    from benchmark import (
        get_numerical_cases, init_numerical_row,
        run_single_numerical_model, compute_live_stats,
        NUMERICAL_MODEL_NAMES,
    )
    from demo_utils import has_llm, LLM_MODEL

    if not has_llm():
        st.warning("Ollama not reachable -- cannot run Llama2 7B benchmark live.")
        return []

    cases = get_numerical_cases()
    models = NUMERICAL_MODEL_NAMES
    labels = NUMERICAL_LABELS
    total_steps = len(cases) * len(models)

    st.subheader(f"Llama2 7B -- Numerical Reasoning (LIVE, model: {LLM_MODEL})")
    st.caption("Architecture: Llama2-7B (7B parameters) -- same for all four approaches")

    # Progress bar
    progress = st.progress(0, text="Starting benchmark...")

    # Live activity log -- shows what's happening right now
    activity_placeholder = st.empty()

    # Accuracy metrics
    metric_cols = st.columns(len(models))
    metric_placeholders = {}
    for col, m in zip(metric_cols, models):
        with col:
            metric_placeholders[m] = st.empty()
            metric_placeholders[m].metric(labels[m], "0%", delta="0/0")

    chart_placeholder = st.empty()
    latency_chart_placeholder = st.empty()

    # Per-case live results table: shows partial results as models complete
    table_placeholder = st.empty()

    # Detailed answers
    answer_container = st.container()

    all_results = []
    current_rows = []  # track partial rows for table display
    start_time = time.perf_counter()
    step = 0

    for i, case in enumerate(cases):
        row = init_numerical_row(case)
        current_rows.append(row)

        for m in models:
            step += 1
            elapsed_so_far = (time.perf_counter() - start_time)
            avg_per_step = elapsed_so_far / step if step > 1 else 0
            remaining_steps = total_steps - step
            eta = f" | ETA: {avg_per_step * remaining_steps:.0f}s" if step > 1 else ""

            short_label = labels[m].split("(")[0].strip()
            progress.progress(
                step / total_steps,
                text=(
                    f"Case {i+1}/{len(cases)} -- "
                    f"Model {models.index(m)+1}/{len(models)}: "
                    f"**{short_label}**{eta}"
                ),
            )

            activity_placeholder.info(
                f"Calling **{short_label}** on: "
                f"*\"{case['question'][:60]}...\"*\n\n"
                f"Expected answer: **{case['expected']}**",
                icon="🔄",
            )

            # Run single model -- this is the slow LLM call
            result = run_single_numerical_model(row, case, m)

            # Immediately show result in activity log
            ok_icon = "✅" if result["correct"] else "❌"
            activity_placeholder.success(
                f"{ok_icon} **{short_label}** returned: "
                f"**{result.get('extracted', '?')}** "
                f"(expected: {case['expected']}) -- "
                f"{result['latency_ms']:.0f}ms",
                icon="✅" if result["correct"] else "❌",
            )

            # Update live results table (shows partial progress per case)
            _update_live_table(table_placeholder, current_rows, models, labels)

        # Case fully complete -- update charts and stats
        all_results.append(row)
        stats = compute_live_stats(all_results, models)

        for m in models:
            s = stats[m]
            metric_placeholders[m].metric(
                labels[m],
                f"{s['accuracy']}%",
                delta=f"{s['correct']}/{s['total']}",
            )

        # Accuracy chart
        fig = go.Figure()
        for m in models:
            s = stats[m]
            fig.add_trace(go.Bar(
                name=labels[m], x=["Accuracy (%)"],
                y=[s["accuracy"]],
                marker_color=COLORS.get(m, "#999"),
                text=[f"{s['accuracy']}%"], textposition="auto",
            ))
        fig.update_layout(
            title=f"Accuracy after {len(all_results)}/{len(cases)} cases",
            barmode="group", yaxis_range=[0, 105],
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Latency chart
        fig2 = go.Figure()
        for m in models:
            s = stats[m]
            if s["avg_latency_ms"]:
                fig2.add_trace(go.Bar(
                    name=labels[m], x=["Avg Latency (ms)"],
                    y=[s["avg_latency_ms"]],
                    marker_color=COLORS.get(m, "#999"),
                    text=[f"{s['avg_latency_ms']:.0f}ms"], textposition="auto",
                ))
        fig2.update_layout(title="Average Latency (running)", barmode="group")
        latency_chart_placeholder.plotly_chart(fig2, use_container_width=True)

        # Show answers for completed case
        with answer_container:
            with st.expander(
                f"Case {i+1}: {case['question'][:60]}... "
                f"(expected: {case['expected']})",
                expanded=(i == len(cases) - 1),
            ):
                for m in models:
                    ok = "CORRECT" if row.get(f"{m}_correct") else "WRONG"
                    ans = row.get(f"{m}_answer", "N/A")[:300]
                    lat = row.get(f"{m}_latency_ms", 0)
                    st.markdown(
                        f"**{labels[m]}** [{ok}] ({lat:.0f}ms):\n> {ans}"
                    )

    elapsed = (time.perf_counter() - start_time) * 1000
    progress.progress(1.0, text=f"Benchmark complete! ({elapsed/1000:.1f}s total)")
    activity_placeholder.success(
        f"All {len(cases)} cases x {len(models)} models complete "
        f"in {elapsed/1000:.1f}s",
        icon="🏁",
    )

    return all_results


def run_live_financial_ratio_benchmark():
    """Run Llama2 7B financial ratio benchmark with per-MODEL live UI updates."""
    import plotly.graph_objects as go
    from benchmark import (
        get_financial_ratio_cases, init_numerical_row,
        run_single_numerical_model, compute_live_stats,
        FINANCIAL_RATIO_MODEL_NAMES,
    )
    from demo_utils import has_llm, LLM_MODEL

    if not has_llm():
        st.warning("Ollama not reachable -- cannot run Financial Ratio benchmark live.")
        return []

    cases = get_financial_ratio_cases()
    models = FINANCIAL_RATIO_MODEL_NAMES
    labels = FINANCIAL_RATIO_LABELS
    total_steps = len(cases) * len(models)

    st.subheader(f"Llama2 7B -- Financial Ratios (LIVE, model: {LLM_MODEL})")
    st.caption("Architecture: Llama2-7B (7B parameters) -- same for all four approaches")

    progress = st.progress(0, text="Starting benchmark...")
    activity_placeholder = st.empty()

    metric_cols = st.columns(len(models))
    metric_placeholders = {}
    for col, m in zip(metric_cols, models):
        with col:
            metric_placeholders[m] = st.empty()
            metric_placeholders[m].metric(labels[m], "0%", delta="0/0")

    chart_placeholder = st.empty()
    latency_chart_placeholder = st.empty()
    table_placeholder = st.empty()
    answer_container = st.container()

    all_results = []
    current_rows = []
    start_time = time.perf_counter()
    step = 0

    for i, case in enumerate(cases):
        row = init_numerical_row(case)
        current_rows.append(row)

        for m in models:
            step += 1
            elapsed_so_far = (time.perf_counter() - start_time)
            avg_per_step = elapsed_so_far / step if step > 1 else 0
            remaining_steps = total_steps - step
            eta = f" | ETA: {avg_per_step * remaining_steps:.0f}s" if step > 1 else ""

            short_label = labels[m].split("(")[0].strip()
            progress.progress(
                step / total_steps,
                text=(
                    f"Case {i+1}/{len(cases)} -- "
                    f"Model {models.index(m)+1}/{len(models)}: "
                    f"**{short_label}**{eta}"
                ),
            )

            activity_placeholder.info(
                f"Calling **{short_label}** on: "
                f"*\"{case['question'][:60]}...\"*\n\n"
                f"Expected answer: **{case['expected']}**",
                icon="🔄",
            )

            result = run_single_numerical_model(row, case, m)

            ok_icon = "✅" if result["correct"] else "❌"
            activity_placeholder.success(
                f"{ok_icon} **{short_label}** returned: "
                f"**{result.get('extracted', '?')}** "
                f"(expected: {case['expected']}) -- "
                f"{result['latency_ms']:.0f}ms",
                icon="✅" if result["correct"] else "❌",
            )

            _update_live_table(table_placeholder, current_rows, models, labels)

        all_results.append(row)
        stats = compute_live_stats(all_results, models)

        for m in models:
            s = stats[m]
            metric_placeholders[m].metric(
                labels[m],
                f"{s['accuracy']}%",
                delta=f"{s['correct']}/{s['total']}",
            )

        fig = go.Figure()
        for m in models:
            s = stats[m]
            fig.add_trace(go.Bar(
                name=labels[m], x=["Accuracy (%)"],
                y=[s["accuracy"]],
                marker_color=COLORS.get(m, "#999"),
                text=[f"{s['accuracy']}%"], textposition="auto",
            ))
        fig.update_layout(
            title=f"Accuracy after {len(all_results)}/{len(cases)} cases",
            barmode="group", yaxis_range=[0, 105],
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        for m in models:
            s = stats[m]
            if s["avg_latency_ms"]:
                fig2.add_trace(go.Bar(
                    name=labels[m], x=["Avg Latency (ms)"],
                    y=[s["avg_latency_ms"]],
                    marker_color=COLORS.get(m, "#999"),
                    text=[f"{s['avg_latency_ms']:.0f}ms"], textposition="auto",
                ))
        fig2.update_layout(title="Average Latency (running)", barmode="group")
        latency_chart_placeholder.plotly_chart(fig2, use_container_width=True)

        with answer_container:
            with st.expander(
                f"Case {i+1}: {case['question'][:60]}... "
                f"(expected: {case['expected']})",
                expanded=(i == len(cases) - 1),
            ):
                for m in models:
                    ok = "CORRECT" if row.get(f"{m}_correct") else "WRONG"
                    ans = row.get(f"{m}_answer", "N/A")[:300]
                    lat = row.get(f"{m}_latency_ms", 0)
                    st.markdown(
                        f"**{labels[m]}** [{ok}] ({lat:.0f}ms):\n> {ans}"
                    )

    elapsed = (time.perf_counter() - start_time) * 1000
    progress.progress(1.0, text=f"Benchmark complete! ({elapsed/1000:.1f}s total)")
    activity_placeholder.success(
        f"All {len(cases)} cases x {len(models)} models complete "
        f"in {elapsed/1000:.1f}s",
        icon="🏁",
    )

    return all_results


def run_live_spam_benchmark():
    """Run DistilBERT 66M spam detection benchmark with live per-case UI updates."""
    import plotly.graph_objects as go
    from benchmark import (
        get_spam_cases, run_single_spam_case,
        compute_live_stats, compute_section_summary,
        SPAM_MODEL_NAMES,
    )

    cases = get_spam_cases()
    models = SPAM_MODEL_NAMES
    labels = SPAM_LABELS

    st.subheader("DistilBERT 66M -- Spam Detection (LIVE)")
    st.caption("Architecture: DistilBERT-base-uncased (66M parameters) -- same for all four approaches")

    # Placeholders for live-updating widgets
    progress = st.progress(0, text="Starting benchmark...")
    metric_cols = st.columns(len(models))
    metric_placeholders = {}
    for col, m in zip(metric_cols, models):
        with col:
            metric_placeholders[m] = st.empty()
            metric_placeholders[m].metric(labels[m], "0%", delta="0/0")

    chart_placeholder = st.empty()
    latency_chart_placeholder = st.empty()
    table_placeholder = st.empty()

    all_results = []
    start_time = time.perf_counter()

    for i, case in enumerate(cases):
        progress.progress(
            (i) / len(cases),
            text=f"Running case {i+1}/{len(cases)}: {case['text'][:50]}..."
        )

        row = run_single_spam_case(case)
        all_results.append(row)

        # Update live stats
        stats = compute_live_stats(all_results, models)

        # Update metrics
        for m in models:
            s = stats[m]
            metric_placeholders[m].metric(
                labels[m],
                f"{s['accuracy']}%",
                delta=f"{s['correct']}/{s['total']}",
            )

        # Update accuracy chart
        fig = go.Figure()
        for m in models:
            s = stats[m]
            fig.add_trace(go.Bar(
                name=labels[m], x=["Accuracy (%)"],
                y=[s["accuracy"]],
                marker_color=COLORS.get(m, "#999"),
                text=[f"{s['accuracy']}%"], textposition="auto",
            ))
        fig.update_layout(
            title=f"Accuracy after {len(all_results)}/{len(cases)} cases",
            barmode="group", yaxis_range=[0, 105],
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Update latency chart
        fig2 = go.Figure()
        for m in models:
            s = stats[m]
            if s["avg_latency_ms"]:
                fig2.add_trace(go.Bar(
                    name=labels[m], x=["Avg Latency (ms)"],
                    y=[s["avg_latency_ms"]],
                    marker_color=COLORS.get(m, "#999"),
                    text=[f"{s['avg_latency_ms']:.0f}ms"], textposition="auto",
                ))
        fig2.update_layout(title="Average Latency (running)", barmode="group")
        latency_chart_placeholder.plotly_chart(fig2, use_container_width=True)

        # Update results table
        table_rows = []
        for r in all_results:
            trow = {
                "Text": r["text"][:55] + "...",
                "Expected": r["expected"].upper(),
            }
            for m in models:
                lbl = r.get(f"{m}_label", "?")
                ok = "Y" if r.get(f"{m}_correct") else "N"
                lat = r.get(f"{m}_latency_ms", 0)
                short = labels[m].split("(")[0].strip()
                trow[short] = f"{lbl} [{ok}] {lat:.0f}ms"
            trow["Category"] = r.get("category", "")
            table_rows.append(trow)
        table_placeholder.dataframe(
            pd.DataFrame(table_rows), use_container_width=True
        )

    elapsed = (time.perf_counter() - start_time) * 1000
    progress.progress(1.0, text=f"Benchmark complete! ({elapsed:.0f}ms total)")

    return all_results


def _update_live_table(placeholder, rows, models, labels):
    """Render the results table, showing partial model results as they come in."""
    table_rows = []
    for r in rows:
        trow = {"Question": r.get("question", "")[:50] + "...",
                "Expected": str(r.get("expected", ""))}
        for m in models:
            short = labels[m].split("(")[0].strip()
            if f"{m}_correct" in r:
                ok = "Y" if r[f"{m}_correct"] else "N"
                extracted = r.get(f"{m}_extracted", "?")
                lat = r.get(f"{m}_latency_ms", 0)
                trow[short] = f"{extracted} [{ok}] {lat:.0f}ms"
            else:
                trow[short] = "..."
        table_rows.append(trow)
    placeholder.dataframe(pd.DataFrame(table_rows), use_container_width=True)


# =========================================================================
# Main tabs
# =========================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "BERT 110M (Sentiment)",
    "Llama2 7B (Numerical)",
    "Llama2 7B (Financial Ratios)",
    "DistilBERT 66M (Spam)",
    "Striking Examples",
])

with tab1:
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_live_sent = st.button(
            "Run Live Benchmark (Sentiment)",
            type="primary", use_container_width=True,
            key="live_sent",
        )
    with col_btn2:
        show_saved_sent = st.button(
            "Show Saved Results",
            use_container_width=True,
            key="saved_sent",
        )

    if run_live_sent:
        sent_results = run_live_sentiment_benchmark()

        # Save results
        from benchmark import compute_section_summary, SENTIMENT_MODEL_NAMES
        sent_summary = compute_section_summary(sent_results, SENTIMENT_MODEL_NAMES)

        # Load existing data to preserve numerical section
        existing = {}
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                existing = json.load(f)

        existing_sections = existing.get("sections", {})
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sections": {
                "bert_110m_sentiment": {
                    "title": "BERT 110M -- Sentiment Classification",
                    "architecture": "BERT-base-uncased (110M parameters)",
                    "models": list(SENTIMENT_MODEL_NAMES),
                    "model_labels": SENTIMENT_LABELS,
                    "summary": sent_summary,
                    "results": sent_results,
                },
                "llama2_7b_numerical": existing_sections.get(
                    "llama2_7b_numerical", {
                        "title": "Llama2 7B -- Numerical Reasoning",
                        "architecture": "Llama2-7B (7B parameters)",
                        "models": ["base", "finetuned", "rag", "hybrid"],
                        "model_labels": NUMERICAL_LABELS,
                        "summary": {},
                        "results": [],
                    }
                ),
                "llama2_7b_financial_ratios": existing_sections.get(
                    "llama2_7b_financial_ratios", {
                        "title": "Llama2 7B -- Financial Ratios",
                        "architecture": "Llama2-7B (7B parameters)",
                        "models": ["base", "finetuned", "rag", "hybrid"],
                        "model_labels": FINANCIAL_RATIO_LABELS,
                        "summary": {},
                        "results": [],
                    }
                ),
                "distilbert_66m_spam": existing_sections.get(
                    "distilbert_66m_spam", {
                        "title": "DistilBERT 66M -- Spam Detection",
                        "architecture": "DistilBERT-base-uncased (66M parameters)",
                        "models": ["base", "finetuned", "rag", "hybrid"],
                        "model_labels": SPAM_LABELS,
                        "summary": {},
                        "results": [],
                    }
                ),
            },
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        st.success("Results saved!")

    elif show_saved_sent or (not run_live_sent):
        if results_data and "bert_110m_sentiment" in results_data.get("sections", {}):
            sections = results_data["sections"]
            timestamp = results_data.get("timestamp", "unknown")
            st.caption(f"Saved results from: {timestamp}")
            render_section(sections["bert_110m_sentiment"])

            s = sections["bert_110m_sentiment"]["summary"]
            base = s.get("base", {})
            fb = s.get("finbert", {})
            rag = s.get("rag", {})
            hyb = s.get("hybrid", {})

            st.divider()
            st.markdown(f"""
            **Analysis (BERT 110M, same architecture):**

            - Base: **{base.get('accuracy',0)}%** | Fine-tuned: **{fb.get('accuracy',0)}%** | RAG: **{rag.get('accuracy',0)}%** | Hybrid: **{hyb.get('accuracy',0)}%**
            - Fine-tuning adds domain knowledge (jargon, context-dependent vocabulary)
            - RAG compensates by retrieving similar examples, but cannot learn new patterns
            - Hybrid combines FinBERT's learned patterns with RAG's retrieval for best coverage
            - Base model has neither advantage and performs worst
            - All four use identical BERT-base 110M architecture -- difference is purely the approach
            """)
        else:
            st.info("No saved results. Click **Run Live Benchmark** to generate them.")

with tab2:
    col_btn3, col_btn4 = st.columns(2)
    with col_btn3:
        run_live_num = st.button(
            "Run Live Benchmark (Numerical)",
            type="primary", use_container_width=True,
            key="live_num",
        )
    with col_btn4:
        show_saved_num = st.button(
            "Show Saved Results",
            use_container_width=True,
            key="saved_num",
        )

    if run_live_num:
        num_results = run_live_numerical_benchmark()

        if num_results:
            from benchmark import compute_section_summary, NUMERICAL_MODEL_NAMES
            num_summary = compute_section_summary(num_results, NUMERICAL_MODEL_NAMES)

            existing = {}
            if RESULTS_PATH.exists():
                with open(RESULTS_PATH) as f:
                    existing = json.load(f)

            existing_sections = existing.get("sections", {})
            output = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sections": {
                    "bert_110m_sentiment": existing_sections.get(
                        "bert_110m_sentiment", {
                            "title": "BERT 110M -- Sentiment Classification",
                            "architecture": "BERT-base-uncased (110M parameters)",
                            "models": ["base", "finbert", "rag", "hybrid"],
                            "model_labels": SENTIMENT_LABELS,
                            "summary": {},
                            "results": [],
                        }
                    ),
                    "llama2_7b_numerical": {
                        "title": "Llama2 7B -- Numerical Reasoning",
                        "architecture": "Llama2-7B (7B parameters)",
                        "models": list(NUMERICAL_MODEL_NAMES),
                        "model_labels": NUMERICAL_LABELS,
                        "summary": num_summary,
                        "results": num_results,
                    },
                    "llama2_7b_financial_ratios": existing_sections.get(
                        "llama2_7b_financial_ratios", {
                            "title": "Llama2 7B -- Financial Ratios",
                            "architecture": "Llama2-7B (7B parameters)",
                            "models": ["base", "finetuned", "rag", "hybrid"],
                            "model_labels": FINANCIAL_RATIO_LABELS,
                            "summary": {},
                            "results": [],
                        }
                    ),
                    "distilbert_66m_spam": existing_sections.get(
                        "distilbert_66m_spam", {
                            "title": "DistilBERT 66M -- Spam Detection",
                            "architecture": "DistilBERT-base-uncased (66M parameters)",
                            "models": ["base", "finetuned", "rag", "hybrid"],
                            "model_labels": SPAM_LABELS,
                            "summary": {},
                            "results": [],
                        }
                    ),
                },
            }
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_PATH, "w") as f:
                json.dump(output, f, indent=2)
            st.success("Results saved!")

    elif show_saved_num or (not run_live_num):
        if results_data and "llama2_7b_numerical" in results_data.get("sections", {}):
            sections = results_data["sections"]
            section = sections["llama2_7b_numerical"]
            timestamp = results_data.get("timestamp", "unknown")
            st.caption(f"Saved results from: {timestamp}")
            if section.get("results"):
                render_section(section)

                s = section["summary"]
                base = s.get("base", {})
                ft = s.get("finetuned", {})
                rag = s.get("rag", {})
                hyb = s.get("hybrid", {})

                st.divider()
                st.markdown(f"""
                **Analysis (Llama2 7B, same architecture):**

                - Base Llama2-7B: **{base.get('accuracy',0)}%** | FinQA-7B: **{ft.get('accuracy',0)}%** | RAG (base): **{rag.get('accuracy',0)}%** | Hybrid (FinQA-7B + RAG): **{hyb.get('accuracy',0)}%**
                - FinQA-7B is Llama2-7B fine-tuned on 8,281 financial QA examples -- same architecture, different weights
                - RAG uses the base Llama2-7B with retrieved documents -- no fine-tuning
                - Hybrid combines FinQA-7B's learned reasoning with RAG's retrieved context
                - All four share the same Llama2-7B (7B parameters) base architecture
                """)

                with st.expander("Full model answers"):
                    for r in section["results"]:
                        st.markdown(f"**Q:** {r['question']}")
                        st.markdown(f"**Expected:** {r['expected']}")
                        for m in section["models"]:
                            ok = "CORRECT" if r.get(f"{m}_correct") else "WRONG"
                            ans = r.get(f"{m}_answer", "N/A")[:300]
                            st.markdown(f"**{section['model_labels'].get(m,m)}** [{ok}]:\n> {ans}")
                        st.divider()
            else:
                st.warning(
                    "Llama2 7B benchmark was skipped (Ollama not available during run). "
                    "Click **Run Live Benchmark** to generate results."
                )
        else:
            st.info("No saved results. Click **Run Live Benchmark** to generate them.")

with tab3:
    col_btn5, col_btn6 = st.columns(2)
    with col_btn5:
        run_live_ratio = st.button(
            "Run Live Benchmark (Financial Ratios)",
            type="primary", use_container_width=True,
            key="live_ratio",
        )
    with col_btn6:
        show_saved_ratio = st.button(
            "Show Saved Results",
            use_container_width=True,
            key="saved_ratio",
        )

    if run_live_ratio:
        ratio_results = run_live_financial_ratio_benchmark()

        if ratio_results:
            from benchmark import compute_section_summary, FINANCIAL_RATIO_MODEL_NAMES
            ratio_summary = compute_section_summary(ratio_results, FINANCIAL_RATIO_MODEL_NAMES)

            existing = {}
            if RESULTS_PATH.exists():
                with open(RESULTS_PATH) as f:
                    existing = json.load(f)

            existing_sections = existing.get("sections", {})
            output = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sections": {
                    "bert_110m_sentiment": existing_sections.get(
                        "bert_110m_sentiment", {
                            "title": "BERT 110M -- Sentiment Classification",
                            "architecture": "BERT-base-uncased (110M parameters)",
                            "models": ["base", "finbert", "rag", "hybrid"],
                            "model_labels": SENTIMENT_LABELS,
                            "summary": {},
                            "results": [],
                        }
                    ),
                    "llama2_7b_numerical": existing_sections.get(
                        "llama2_7b_numerical", {
                            "title": "Llama2 7B -- Numerical Reasoning",
                            "architecture": "Llama2-7B (7B parameters)",
                            "models": ["base", "finetuned", "rag", "hybrid"],
                            "model_labels": NUMERICAL_LABELS,
                            "summary": {},
                            "results": [],
                        }
                    ),
                    "llama2_7b_financial_ratios": {
                        "title": "Llama2 7B -- Financial Ratios",
                        "architecture": "Llama2-7B (7B parameters)",
                        "models": list(FINANCIAL_RATIO_MODEL_NAMES),
                        "model_labels": FINANCIAL_RATIO_LABELS,
                        "summary": ratio_summary,
                        "results": ratio_results,
                    },
                    "distilbert_66m_spam": existing_sections.get(
                        "distilbert_66m_spam", {
                            "title": "DistilBERT 66M -- Spam Detection",
                            "architecture": "DistilBERT-base-uncased (66M parameters)",
                            "models": ["base", "finetuned", "rag", "hybrid"],
                            "model_labels": SPAM_LABELS,
                            "summary": {},
                            "results": [],
                        }
                    ),
                },
            }
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_PATH, "w") as f:
                json.dump(output, f, indent=2)
            st.success("Results saved!")

    elif show_saved_ratio or (not run_live_ratio):
        if results_data and "llama2_7b_financial_ratios" in results_data.get("sections", {}):
            sections = results_data["sections"]
            section = sections["llama2_7b_financial_ratios"]
            timestamp = results_data.get("timestamp", "unknown")
            st.caption(f"Saved results from: {timestamp}")
            if section.get("results"):
                render_section(section)

                s = section["summary"]
                base = s.get("base", {})
                ft = s.get("finetuned", {})
                rag = s.get("rag", {})
                hyb = s.get("hybrid", {})

                st.divider()

                # Determine best model dynamically
                accuracies = {"Base Llama2-7B": base.get('accuracy', 0),
                              "FinQA-7B": ft.get('accuracy', 0),
                              "RAG (base)": rag.get('accuracy', 0),
                              "Hybrid (FinQA-7B + RAG)": hyb.get('accuracy', 0)}
                best_model = max(accuracies, key=accuracies.get)
                best_acc = accuracies[best_model]

                st.markdown(f"""
                **Analysis (Llama2 7B, Financial Ratios):**

                - Base Llama2-7B: **{base.get('accuracy',0)}%** | FinQA-7B: **{ft.get('accuracy',0)}%** | RAG (base): **{rag.get('accuracy',0)}%** | Hybrid (FinQA-7B + RAG): **{hyb.get('accuracy',0)}%**
                - **Best performer: {best_model} at {best_acc}%**
                - These multi-step ratio calculations test formula knowledge, chained arithmetic, and domain conventions
                - Fine-tuning helps most on tasks requiring domain-specific reasoning chains (e.g., DuPont decomposition, sustainable growth rate)
                - RAG adds latency and can introduce noise when retrieved documents are not directly relevant to the calculation
                - Performance varies by category -- see the category breakdown for details
                - All four share the same Llama2-7B (7B parameters) base architecture
                """)

                with st.expander("Full model answers"):
                    for r in section["results"]:
                        st.markdown(f"**Q:** {r['question']}")
                        st.markdown(f"**Expected:** {r['expected']}")
                        for m in section["models"]:
                            ok = "CORRECT" if r.get(f"{m}_correct") else "WRONG"
                            ans = r.get(f"{m}_answer", "N/A")[:300]
                            st.markdown(f"**{section['model_labels'].get(m,m)}** [{ok}]:\n> {ans}")
                        st.divider()
            else:
                st.warning(
                    "Financial Ratio benchmark was skipped (Ollama not available during run). "
                    "Click **Run Live Benchmark** to generate results."
                )
        else:
            st.info("No saved results. Click **Run Live Benchmark** to generate them.")

with tab4:
    col_btn7, col_btn8 = st.columns(2)
    with col_btn7:
        run_live_spam = st.button(
            "Run Live Benchmark (Spam Detection)",
            type="primary", use_container_width=True,
            key="live_spam",
        )
    with col_btn8:
        show_saved_spam = st.button(
            "Show Saved Results",
            use_container_width=True,
            key="saved_spam",
        )

    if run_live_spam:
        spam_results = run_live_spam_benchmark()

        # Save results
        from benchmark import compute_section_summary, SPAM_MODEL_NAMES
        spam_summary = compute_section_summary(spam_results, SPAM_MODEL_NAMES)

        # Load existing data to preserve other sections
        existing = {}
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                existing = json.load(f)

        existing_sections = existing.get("sections", {})
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sections": {
                "bert_110m_sentiment": existing_sections.get(
                    "bert_110m_sentiment", {
                        "title": "BERT 110M -- Sentiment Classification",
                        "architecture": "BERT-base-uncased (110M parameters)",
                        "models": ["base", "finbert", "rag", "hybrid"],
                        "model_labels": SENTIMENT_LABELS,
                        "summary": {},
                        "results": [],
                    }
                ),
                "llama2_7b_numerical": existing_sections.get(
                    "llama2_7b_numerical", {
                        "title": "Llama2 7B -- Numerical Reasoning",
                        "architecture": "Llama2-7B (7B parameters)",
                        "models": ["base", "finetuned", "rag", "hybrid"],
                        "model_labels": NUMERICAL_LABELS,
                        "summary": {},
                        "results": [],
                    }
                ),
                "llama2_7b_financial_ratios": existing_sections.get(
                    "llama2_7b_financial_ratios", {
                        "title": "Llama2 7B -- Financial Ratios",
                        "architecture": "Llama2-7B (7B parameters)",
                        "models": ["base", "finetuned", "rag", "hybrid"],
                        "model_labels": FINANCIAL_RATIO_LABELS,
                        "summary": {},
                        "results": [],
                    }
                ),
                "distilbert_66m_spam": {
                    "title": "DistilBERT 66M -- Spam Detection",
                    "architecture": "DistilBERT-base-uncased (66M parameters)",
                    "models": list(SPAM_MODEL_NAMES),
                    "model_labels": SPAM_LABELS,
                    "summary": spam_summary,
                    "results": spam_results,
                },
            },
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        st.success("Results saved!")

    elif show_saved_spam or (not run_live_spam):
        if results_data and "distilbert_66m_spam" in results_data.get("sections", {}):
            sections = results_data["sections"]
            timestamp = results_data.get("timestamp", "unknown")
            st.caption(f"Saved results from: {timestamp}")
            render_section(sections["distilbert_66m_spam"])

            s = sections["distilbert_66m_spam"]["summary"]
            base = s.get("base", {})
            ft = s.get("finetuned", {})
            rag = s.get("rag", {})
            hyb = s.get("hybrid", {})

            st.divider()
            st.markdown(f"""
            **Analysis (DistilBERT 66M, same architecture):**

            - Base: **{base.get('accuracy',0)}%** | Fine-tuned: **{ft.get('accuracy',0)}%** | RAG: **{rag.get('accuracy',0)}%** | Hybrid: **{hyb.get('accuracy',0)}%**
            - Fine-tuning teaches the model patterns of spam (urgency, phishing language, suspicious URLs)
            - RAG retrieves similar labeled examples for comparison-based classification
            - Hybrid combines fine-tuned classification with RAG retrieval for best coverage
            - Base model uses zero-shot cosine similarity with no domain knowledge
            - All four use identical DistilBERT-base 66M architecture -- difference is purely the approach
            """)
        else:
            st.info("No saved results. Click **Run Live Benchmark** to generate them.")

with tab5:
    st.subheader("Striking Examples")

    with open(TEST_CASES_PATH) as f:
        striking = json.load(f).get("striking_examples", {})

    st.markdown("### Where Fine-Tuning Wins")
    for ex in striking.get("finetuning_wins", []):
        with st.expander(f"{ex['text'][:70]}..."):
            st.markdown(f"**Text:** *{ex['text']}*")
            st.markdown(f"**Correct:** {ex['label'].upper()}")
            st.markdown(f"**Why:** {ex['why']}")

    st.markdown("### Where RAG Wins")
    for ex in striking.get("rag_wins", []):
        with st.expander(f"{ex['question'][:70]}..."):
            st.markdown(f"**Question:** *{ex['question']}*")
            st.markdown(f"**Why:** {ex['why']}")

    st.markdown("### Where Hybrid Wins")
    for ex in striking.get("hybrid_wins", []):
        with st.expander(f"{ex['question'][:70]}..."):
            st.markdown(f"**Question:** *{ex['question']}*")
            st.markdown(f"**Why:** {ex['why']}")

st.divider()
st.markdown(
    "All results measured in this environment. Four controlled experiments, "
    "each comparing four approaches (base, fine-tuned, RAG, hybrid) while "
    "keeping architecture constant."
)
