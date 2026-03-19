"""
Benchmark Model Family -- Does model size matter for fine-tuning?

Compares two fine-tuned models on the SAME spam-detection task:
  - Fine-tuned DistilBERT  (66M parameters, local)
  - Fine-tuned GPT-4o-mini (~8B parameters, OpenAI API)

Both trained on the same spam dataset. Evaluated on basic and adversarial cases.
"""
import streamlit as st
import pandas as pd
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Benchmark Model Family", page_icon="FT", layout="wide")

st.title("Benchmark Model Family")
st.markdown("""
**Does a bigger fine-tuned model beat a smaller one on the same task?**

Both models were fine-tuned on the **same spam-detection training dataset**.
The only difference is the base model size and architecture.

| Model | Base | Parameters | Inference | Cost |
|-------|------|-----------|-----------|------|
| **Fine-tuned DistilBERT** | distilbert-base-uncased | **66M** | Local CPU/GPU | ~$0.01/1M tokens |
| **Fine-tuned GPT-4o-mini** | gpt-4o-mini | **~8B** (undisclosed) | OpenAI API | $0.30/$1.20 per 1M in/out |

Size ratio: **~121x larger**. Does 121x more parameters = better spam detection?
""")

RESULTS_PATH = Path(__file__).parent.parent.parent / "data" / "model_family_results.json"

COLORS = {
    "distilbert_ft": "#ff6b35",   # orange for small model
    "gpt4omini_ft": "#4a90d9",    # blue for large model
}

MODEL_LABELS = {
    "distilbert_ft": "Fine-tuned DistilBERT (66M)",
    "gpt4omini_ft": "Fine-tuned GPT-4o-mini (~8B)",
}

# Load results
results_data = None
if RESULTS_PATH.exists():
    try:
        with open(RESULTS_PATH) as f:
            results_data = json.load(f)
        if "sections" not in results_data:
            results_data = None
    except (json.JSONDecodeError, OSError):
        results_data = None


# =========================================================================
# Rendering helpers
# =========================================================================
def render_model_cards():
    """Render model info comparison cards."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="border: 2px solid #ff6b35; border-radius: 10px; padding: 15px; text-align: center;">
        <h3 style="color: #ff6b35;">Fine-tuned DistilBERT</h3>
        <p style="font-size: 2em; font-weight: bold;">66M params</p>
        <p>Local inference | Near-zero cost<br/>
        distilbert-base-uncased<br/>
        Custom HuggingFace fine-tuning</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="border: 2px solid #4a90d9; border-radius: 10px; padding: 15px; text-align: center;">
        <h3 style="color: #4a90d9;">Fine-tuned GPT-4o-mini</h3>
        <p style="font-size: 2em; font-weight: bold;">~8B params</p>
        <p>OpenAI API | $0.30/$1.20 per 1M tokens<br/>
        gpt-4o-mini-2024-07-18<br/>
        OpenAI fine-tuning platform</p>
        </div>
        """, unsafe_allow_html=True)


def render_section(section_data, section_key):
    """Render a benchmark section with charts and tables."""
    import plotly.graph_objects as go

    title = section_data["title"]
    models = section_data["models"]
    labels = section_data["model_labels"]
    summary = section_data["summary"]
    per_example = section_data["results"]

    if not per_example:
        st.info("No results available for this section.")
        return

    st.subheader(title)
    if section_data.get("subtitle"):
        st.caption(section_data["subtitle"])

    # ----- Accuracy metrics -----
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        s = summary.get(m, {})
        with col:
            st.metric(labels.get(m, m),
                      f"{s.get('accuracy', 0)}%",
                      delta=f"{s.get('correct', 0)}/{s.get('total', 0)}")

    # ----- Accuracy comparison chart -----
    fig_acc = go.Figure()
    for m in models:
        s = summary.get(m, {})
        fig_acc.add_trace(go.Bar(
            name=labels.get(m, m),
            x=["Accuracy (%)"],
            y=[s.get("accuracy", 0)],
            marker_color=COLORS.get(m, "#999"),
            text=[f"{s.get('accuracy', 0)}%"],
            textposition="auto",
        ))
    fig_acc.update_layout(
        title="Accuracy Comparison",
        barmode="group",
        yaxis_range=[0, 105],
    )
    st.plotly_chart(fig_acc, use_container_width=True, key=f"acc_{section_key}")

    # ----- Multi-metric comparison (accuracy, latency, cost) -----
    col_lat, col_cost = st.columns(2)

    with col_lat:
        fig_lat = go.Figure()
        for m in models:
            s = summary.get(m, {})
            lat = s.get("avg_latency_ms", 0)
            if lat:
                fig_lat.add_trace(go.Bar(
                    name=labels.get(m, m),
                    x=["Avg Latency (ms)"],
                    y=[lat],
                    marker_color=COLORS.get(m, "#999"),
                    text=[f"{lat:.0f}ms"],
                    textposition="auto",
                ))
        fig_lat.update_layout(title="Average Latency", barmode="group")
        st.plotly_chart(fig_lat, use_container_width=True, key=f"lat_{section_key}")

    with col_cost:
        fig_cost = go.Figure()
        for m in models:
            s = summary.get(m, {})
            cost = s.get("cost_per_1k_queries_usd", 0)
            fig_cost.add_trace(go.Bar(
                name=labels.get(m, m),
                x=["Cost per 1K queries ($)"],
                y=[cost],
                marker_color=COLORS.get(m, "#999"),
                text=[f"${cost:.4f}"],
                textposition="auto",
            ))
        fig_cost.update_layout(title="Cost per 1,000 Queries", barmode="group")
        st.plotly_chart(fig_cost, use_container_width=True, key=f"cost_{section_key}")

    # ----- F1 Score -----
    f1_data = {m: summary.get(m, {}).get("f1_macro") for m in models}
    if any(v is not None for v in f1_data.values()):
        st.markdown("##### F1 Score (Macro)")
        f1_cols = st.columns(len(models))
        for col, m in zip(f1_cols, models):
            s = summary.get(m, {})
            with col:
                f1 = s.get("f1_macro", 0)
                prec = s.get("precision_macro", 0)
                rec = s.get("recall_macro", 0)
                st.metric(labels.get(m, m), f"F1: {f1:.3f}")
                st.caption(f"Precision: {prec:.3f} / Recall: {rec:.3f}")

        # F1 bar chart
        fig_f1 = go.Figure()
        for m in models:
            s = summary.get(m, {})
            for metric, label_suffix in [("f1_macro", "F1"), ("precision_macro", "Precision"), ("recall_macro", "Recall")]:
                val = s.get(metric, 0)
                fig_f1.add_trace(go.Bar(
                    name=f"{labels.get(m, m)} - {label_suffix}",
                    x=[label_suffix],
                    y=[val],
                    marker_color=COLORS.get(m, "#999"),
                    text=[f"{val:.3f}"],
                    textposition="auto",
                    opacity=1.0 if metric == "f1_macro" else 0.7,
                ))
        fig_f1.update_layout(title="F1 / Precision / Recall", barmode="group",
                             yaxis_range=[0, 1.05])
        st.plotly_chart(fig_f1, use_container_width=True, key=f"f1_{section_key}")

    # ----- Per-class F1 breakdown -----
    for m in models:
        per_class = summary.get(m, {}).get("f1_per_class")
        if per_class:
            with st.expander(f"Per-class F1 breakdown -- {labels.get(m, m)}"):
                pc_rows = []
                for cls, metrics in per_class.items():
                    pc_rows.append({
                        "Class": cls.upper(),
                        "Precision": f"{metrics['precision']:.3f}",
                        "Recall": f"{metrics['recall']:.3f}",
                        "F1": f"{metrics['f1']:.3f}",
                    })
                st.table(pd.DataFrame(pc_rows))

    # ----- Token usage -----
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
                total_cost = s.get("total_cost_usd", 0)
                st.metric(labels.get(m, m).split("(")[0].strip(),
                          f"{avg_tok:,} tok/query")
                st.caption(f"Cost/1K queries: ${cost_1k:.4f}")
                st.caption(f"Total cost this run: ${total_cost:.6f}")

    # ----- Category breakdown -----
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
        st.markdown("##### Accuracy by Category")
        st.table(pd.DataFrame(cat_rows))

        # Category chart
        cats = [r["Category"] for r in cat_rows]
        fig_cat = go.Figure()
        for m in models:
            short = labels.get(m, m).split("(")[0].strip()
            accs = [float(r[short].replace("%", "")) for r in cat_rows]
            fig_cat.add_trace(go.Bar(
                name=labels.get(m, m), x=cats, y=accs,
                marker_color=COLORS.get(m, "#999"),
            ))
        fig_cat.update_layout(title="Accuracy by Category", barmode="group",
                              yaxis_range=[0, 105])
        st.plotly_chart(fig_cat, use_container_width=True, key=f"cat_{section_key}")

    # ----- LLM Judge scores -----
    if results_data and results_data.get("with_judge"):
        judge_sums = results_data.get("judge_summaries", {}).get(section_key, {})
        if judge_sums:
            st.markdown("---")
            st.markdown("##### LLM-as-Judge Evaluation")
            judge_cols = st.columns(len(models))
            for col, m in zip(judge_cols, models):
                js = judge_sums.get(m, {})
                with col:
                    if js.get("count", 0) > 0:
                        st.metric(labels.get(m, m),
                                  f"Overall: {js['overall']:.1f}/5")
                        st.caption(
                            f"Correctness: {js['correctness']:.1f} | "
                            f"Reasoning: {js['reasoning_quality']:.1f} | "
                            f"Faithfulness: {js['faithfulness']:.1f}"
                        )
                    else:
                        st.metric(labels.get(m, m), "N/A")

            # Judge radar chart
            fig_radar = go.Figure()
            categories_radar = ["Correctness", "Reasoning Quality", "Faithfulness"]
            for m in models:
                js = judge_sums.get(m, {})
                if js.get("count", 0) > 0:
                    vals = [js["correctness"], js["reasoning_quality"], js["faithfulness"]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=categories_radar + [categories_radar[0]],
                        fill="toself",
                        name=labels.get(m, m),
                        line_color=COLORS.get(m, "#999"),
                    ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                title="Judge Evaluation Radar",
            )
            st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{section_key}")

    # ----- Per-example results table -----
    with st.expander("Per-example results"):
        rows = []
        for r in per_example:
            row = {
                "Text": r["text"][:60] + "...",
                "Expected": r.get("expected", "").upper(),
            }
            for m in models:
                lbl = r.get(f"{m}_label", "?")
                ok = "Y" if r.get(f"{m}_correct") else "N"
                conf = r.get(f"{m}_confidence", 0)
                lat = r.get(f"{m}_latency_ms", 0)
                short = labels.get(m, m).split("(")[0].strip()
                row[short] = f"{lbl} [{ok}] {conf:.2f} {lat:.0f}ms"

                # Judge overall if available
                judge = r.get(f"{m}_judge")
                if judge:
                    row[f"{short} Judge"] = f"{judge.get('overall', 0):.1f}/5"
            row["Category"] = r.get("category", "")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_combined_analysis():
    """Render combined analysis across both sections."""
    import plotly.graph_objects as go

    if not results_data:
        return

    sections = results_data.get("sections", {})
    basic = sections.get("basic_spam", {})
    adv = sections.get("adversarial_spam", {})

    if not basic.get("results") and not adv.get("results"):
        return

    st.markdown("---")
    st.header("Combined Analysis: Does Size Matter?")

    models = MODEL_LABELS.keys()

    # Gather accuracy across sections
    rows = []
    for section_key, section_label in [("basic_spam", "Basic (20 cases)"),
                                        ("adversarial_spam", "Adversarial (30 cases)")]:
        section = sections.get(section_key, {})
        summary = section.get("summary", {})
        if not summary:
            continue
        for m in models:
            s = summary.get(m, {})
            rows.append({
                "Model": MODEL_LABELS[m],
                "Test Suite": section_label,
                "Accuracy (%)": s.get("accuracy", 0),
                "F1 (Macro)": s.get("f1_macro", 0),
                "Avg Latency (ms)": s.get("avg_latency_ms", 0),
                "Cost/1K ($)": s.get("cost_per_1k_queries_usd", 0),
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

    # Combined accuracy chart
    fig = go.Figure()
    for m in models:
        accs = []
        x_labels = []
        for section_key, section_label in [("basic_spam", "Basic"),
                                            ("adversarial_spam", "Adversarial")]:
            section = sections.get(section_key, {})
            summary = section.get("summary", {})
            if summary:
                accs.append(summary.get(m, {}).get("accuracy", 0))
                x_labels.append(section_label)
        if accs:
            fig.add_trace(go.Bar(
                name=MODEL_LABELS[m],
                x=x_labels,
                y=accs,
                marker_color=COLORS.get(m, "#999"),
                text=[f"{a}%" for a in accs],
                textposition="auto",
            ))
    fig.update_layout(
        title="Accuracy: Basic vs Adversarial",
        barmode="group",
        yaxis_range=[0, 105],
        yaxis_title="Accuracy (%)",
    )
    st.plotly_chart(fig, use_container_width=True, key="combined_acc")

    # Accuracy drop analysis
    basic_summary = basic.get("summary", {})
    adv_summary = adv.get("summary", {})
    if basic_summary and adv_summary:
        st.markdown("##### Adversarial Robustness (accuracy drop from basic to adversarial)")
        drop_cols = st.columns(len(list(models)))
        for col, m in zip(drop_cols, models):
            basic_acc = basic_summary.get(m, {}).get("accuracy", 0)
            adv_acc = adv_summary.get(m, {}).get("accuracy", 0)
            drop = basic_acc - adv_acc
            with col:
                st.metric(
                    MODEL_LABELS[m],
                    f"{adv_acc}%",
                    delta=f"{-drop:+.1f}% from basic",
                    delta_color="inverse",
                )

    # Cost-efficiency analysis
    st.markdown("##### Cost-Efficiency Analysis")
    efficiency_rows = []
    for section_key, section_label in [("basic_spam", "Basic"),
                                        ("adversarial_spam", "Adversarial")]:
        section = sections.get(section_key, {})
        summary = section.get("summary", {})
        if not summary:
            continue
        for m in models:
            s = summary.get(m, {})
            acc = s.get("accuracy", 0)
            cost = s.get("cost_per_1k_queries_usd", 0)
            # Accuracy per dollar (higher is better)
            acc_per_dollar = (acc / cost) if cost > 0 else float('inf')
            efficiency_rows.append({
                "Model": MODEL_LABELS[m],
                "Test Suite": section_label,
                "Accuracy": f"{acc}%",
                "Cost/1K Queries": f"${cost:.4f}",
                "Accuracy/$ (higher=better)": f"{acc_per_dollar:,.0f}" if acc_per_dollar != float('inf') else "near-zero cost",
            })
    if efficiency_rows:
        st.table(pd.DataFrame(efficiency_rows))

    # Key findings
    st.markdown("##### Key Findings")

    findings = []
    if basic_summary and adv_summary:
        db_basic = basic_summary.get("distilbert_ft", {}).get("accuracy", 0)
        gpt_basic = adv_summary.get("gpt4omini_ft", {}).get("accuracy", 0)
        db_adv = adv_summary.get("distilbert_ft", {}).get("accuracy", 0)
        gpt_adv = adv_summary.get("gpt4omini_ft", {}).get("accuracy", 0)
        gpt_basic_real = basic_summary.get("gpt4omini_ft", {}).get("accuracy", 0)

        if gpt_basic_real > db_basic:
            findings.append(f"GPT-4o-mini outperforms DistilBERT on basic cases ({gpt_basic_real}% vs {db_basic}%)")
        elif db_basic > gpt_basic_real:
            findings.append(f"DistilBERT outperforms GPT-4o-mini on basic cases ({db_basic}% vs {gpt_basic_real}%)")
        else:
            findings.append(f"Both models tied on basic cases at {db_basic}%")

        if gpt_adv > db_adv:
            findings.append(f"GPT-4o-mini is more robust on adversarial cases ({gpt_adv}% vs {db_adv}%)")
        elif db_adv > gpt_adv:
            findings.append(f"DistilBERT is more robust on adversarial cases ({db_adv}% vs {gpt_adv}%)")

        db_drop = basic_summary.get("distilbert_ft", {}).get("accuracy", 0) - db_adv
        gpt_drop = gpt_basic_real - gpt_adv
        if abs(db_drop - gpt_drop) > 2:
            less_drop = "DistilBERT" if db_drop < gpt_drop else "GPT-4o-mini"
            findings.append(f"{less_drop} shows less accuracy degradation under adversarial conditions")

        db_cost = basic_summary.get("distilbert_ft", {}).get("cost_per_1k_queries_usd", 0)
        gpt_cost = basic_summary.get("gpt4omini_ft", {}).get("cost_per_1k_queries_usd", 0)
        if gpt_cost > 0 and db_cost > 0:
            cost_ratio = gpt_cost / db_cost
            findings.append(f"GPT-4o-mini costs ~{cost_ratio:.0f}x more per query than DistilBERT")
        elif gpt_cost > 0:
            findings.append(f"GPT-4o-mini costs ${gpt_cost:.4f}/1K queries vs near-zero for DistilBERT")

    if findings:
        for f in findings:
            st.markdown(f"- {f}")
    else:
        st.info("Run the benchmark to generate findings.")


def run_live_benchmark():
    """Run the model family benchmark with live UI updates."""
    import plotly.graph_objects as go

    from model_family_benchmark import (
        MODEL_NAMES, MODEL_LABELS as MF_LABELS,
        get_basic_spam_cases, get_adversarial_spam_cases,
        run_single_case, has_openai, _estimate_family_cost,
    )
    from benchmark import compute_section_summary

    if not has_openai():
        st.error("OPENAI_API_KEY not set. Cannot call GPT-4o-mini fine-tuned model.")
        return

    # Check for judge
    judge_model = None
    with_judge = st.session_state.get("mf_with_judge", False)
    if with_judge:
        from llm_judge import get_judge_model_name
        judge_model = get_judge_model_name()
        if not judge_model:
            st.warning("No judge model available, running without judge")
            with_judge = False

    models = MODEL_NAMES
    labels = MF_LABELS

    all_sections = {
        "basic": ("Basic Spam Detection", get_basic_spam_cases()),
        "adversarial": ("Adversarial Spam Detection", get_adversarial_spam_cases()),
    }

    all_results = {}

    for section_key, (section_title, cases) in all_sections.items():
        st.subheader(f"{section_title} (LIVE)")
        progress = st.progress(0, text=f"Starting {section_title}...")

        metric_cols = st.columns(len(models))
        metric_placeholders = {}
        for col, m in zip(metric_cols, models):
            with col:
                metric_placeholders[m] = st.empty()
                metric_placeholders[m].metric(labels[m], "0%", delta="0/0")

        chart_placeholder = st.empty()
        table_placeholder = st.empty()

        results = []
        start_time = time.perf_counter()

        for i, case in enumerate(cases):
            progress.progress(
                i / len(cases),
                text=f"[{section_title}] Case {i+1}/{len(cases)}: {case['text'][:50]}..."
            )

            row = run_single_case(case, with_judge=with_judge, judge_model=judge_model)
            results.append(row)

            # Update metrics
            total = len(results)
            for m in models:
                correct = sum(1 for r in results if r.get(f"{m}_correct"))
                acc = round(correct / total * 100, 1)
                metric_placeholders[m].metric(labels[m], f"{acc}%",
                                              delta=f"{correct}/{total}")

            # Update chart
            fig = go.Figure()
            for m in models:
                correct = sum(1 for r in results if r.get(f"{m}_correct"))
                acc = round(correct / total * 100, 1)
                fig.add_trace(go.Bar(
                    name=labels[m], x=["Accuracy (%)"],
                    y=[acc],
                    marker_color=COLORS.get(m, "#999"),
                    text=[f"{acc}%"], textposition="auto",
                ))
            fig.update_layout(
                title=f"Accuracy after {total}/{len(cases)} cases",
                barmode="group", yaxis_range=[0, 105],
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            # Update table
            table_rows = []
            for r in results:
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
            table_placeholder.dataframe(pd.DataFrame(table_rows),
                                        use_container_width=True)

        elapsed = (time.perf_counter() - start_time) * 1000
        progress.progress(1.0, text=f"{section_title} complete! ({elapsed/1000:.1f}s)")
        all_results[section_key] = results

    # Save all results
    from model_family_benchmark import _orig_save
    judge_summaries = {}
    if with_judge:
        from llm_judge import compute_judge_summary
        for key, results in all_results.items():
            if results:
                judge_summaries[key + "_spam"] = compute_judge_summary(results, models)

    _orig_save(
        basic_results=all_results.get("basic"),
        adversarial_results=all_results.get("adversarial"),
        with_judge=with_judge,
        judge_summaries=judge_summaries,
    )
    st.success("Results saved!")


# =========================================================================
# Main page layout
# =========================================================================
render_model_cards()
st.markdown("---")

# Controls
col_run, col_judge, col_saved = st.columns([2, 1, 1])
with col_run:
    run_live = st.button(
        "Run Live Benchmark",
        type="primary", use_container_width=True,
        key="mf_run_live",
    )
with col_judge:
    with_judge = st.checkbox("With LLM Judge", key="mf_with_judge")
with col_saved:
    show_saved = st.button(
        "Show Saved Results",
        use_container_width=True,
        key="mf_show_saved",
    )

if run_live:
    run_live_benchmark()

elif show_saved or (not run_live):
    if results_data:
        timestamp = results_data.get("timestamp", "unknown")
        st.caption(f"Saved results from: {timestamp}")
        if results_data.get("with_judge"):
            st.caption("Includes LLM-as-Judge evaluation")

        sections = results_data.get("sections", {})

        tab_basic, tab_adv, tab_analysis = st.tabs([
            "Basic Test Cases",
            "Adversarial Test Cases",
            "Combined Analysis",
        ])

        with tab_basic:
            if "basic_spam" in sections and sections["basic_spam"].get("results"):
                render_section(sections["basic_spam"], "basic_spam")
            else:
                st.info("No basic test case results. Click **Run Live Benchmark** to generate them.")

        with tab_adv:
            if "adversarial_spam" in sections and sections["adversarial_spam"].get("results"):
                render_section(sections["adversarial_spam"], "adversarial_spam")
            else:
                st.info("No adversarial results. Click **Run Live Benchmark** to generate them.")

        with tab_analysis:
            render_combined_analysis()
    else:
        st.info("No saved results found. Click **Run Live Benchmark** to generate them.")

st.divider()
st.markdown(
    "**Model Family Benchmark** -- Comparing fine-tuned models of different sizes "
    "(66M vs ~8B parameters) on the same spam detection task with the same training data. "
    "All results measured in this environment."
)
