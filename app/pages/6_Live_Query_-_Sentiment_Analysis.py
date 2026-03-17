"""
Live Query - Sentiment Analysis
Compare Base BERT vs FinBERT vs BERT+RAG vs Hybrid on any financial sentence.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils import (
    run_finbert, run_base_bert_sentiment, run_rag_sentiment,
    run_hybrid_sentiment, get_demo_status,
)

st.set_page_config(page_title="Live Query - Sentiment", page_icon="FT", layout="wide")

st.title("Live Query: Sentiment Analysis")
st.markdown(
    "Enter a financial sentence and compare all four approaches side-by-side. "
    "All models use **BERT 110M parameters** -- the difference is fine-tuning and retrieval."
)

with st.expander("Comparison Methodology"):
    st.markdown("""
    | Approach | Model | What changes |
    |----------|-------|-------------|
    | **Base BERT** | bert-base-uncased (110M) | Nothing -- generic model |
    | **FinBERT** | ProsusAI/finbert (110M) | Fine-tuned on Financial PhraseBank |
    | **BERT + RAG** | bert-base-uncased + ChromaDB | Retrieval of similar labeled sentences, majority vote |
    | **Hybrid** | FinBERT + RAG | FinBERT prediction weighted with RAG neighbor vote |
    """)

# Sidebar
with st.sidebar:
    st.header("Status")
    status = get_demo_status()
    for k, v in status.items():
        if v == "live":
            st.success(f"[LIVE] {k}")
        else:
            st.error(f"[OFFLINE] {k}")

# Sample sentences
SAMPLES = [
    "Net interest income grew 12% driven by higher rates and loan growth.",
    "Management expects headwinds from deposit competition to persist throughout 2024.",
    "The company maintained its quarterly dividend of $0.50 per share.",
    "Operating efficiency improved with the cost-to-income ratio declining to 52%.",
    "Provisions for loan losses rose sharply amid deteriorating credit quality.",
    "The board approved a routine extension of the existing credit facility.",
    "Margin compression accelerated due to competitive deposit pricing pressures.",
]

with st.form("sentiment_form"):
    text = st.text_area(
        "Financial sentence:",
        value=SAMPLES[0],
        height=80,
    )
    st.caption("Try these examples:")
    for i, sample in enumerate(SAMPLES[1:], 1):
        st.caption(f"{i}. {sample}")
    submitted = st.form_submit_button("Analyze Sentiment", type="primary",
                                       use_container_width=True)

if submitted and text.strip():
    st.divider()

    col_base, col_ft, col_rag, col_hyb = st.columns(4)

    with col_base:
        st.subheader("Base BERT")
        with st.spinner("Running..."):
            try:
                r = run_base_bert_sentiment(text)
                color = {"positive": "green", "negative": "red", "neutral": "orange"}.get(r.label, "gray")
                st.markdown(f"### :{color}[{r.label.upper()}]")
                st.metric("Confidence", f"{r.confidence:.1%}")
                st.caption(f"Latency: {r.latency_ms:.0f}ms")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_ft:
        st.subheader("FinBERT")
        st.caption("Fine-tuned")
        with st.spinner("Running..."):
            try:
                r = run_finbert(text)
                color = {"positive": "green", "negative": "red", "neutral": "orange"}.get(r.label, "gray")
                st.markdown(f"### :{color}[{r.label.upper()}]")
                st.metric("Confidence", f"{r.confidence:.1%}")
                st.caption(f"Latency: {r.latency_ms:.0f}ms")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_rag:
        st.subheader("BERT + RAG")
        with st.spinner("Running..."):
            try:
                r = run_rag_sentiment(text)
                color = {"positive": "green", "negative": "red", "neutral": "orange"}.get(r.label, "gray")
                st.markdown(f"### :{color}[{r.label.upper()}]")
                st.metric("Confidence", f"{r.confidence:.1%}")
                st.caption(f"Latency: {r.latency_ms:.0f}ms")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_hyb:
        st.subheader("Hybrid")
        st.caption("FinBERT + RAG")
        with st.spinner("Running..."):
            try:
                r = run_hybrid_sentiment(text)
                color = {"positive": "green", "negative": "red", "neutral": "orange"}.get(r.label, "gray")
                st.markdown(f"### :{color}[{r.label.upper()}]")
                st.metric("Confidence", f"{r.confidence:.1%}")
                st.caption(f"Latency: {r.latency_ms:.0f}ms")
            except Exception as e:
                st.error(f"Error: {e}")
