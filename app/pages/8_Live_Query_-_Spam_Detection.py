"""
Live Query - Spam Detection
Compare Base DistilBERT vs Fine-Tuned vs DistilBERT+RAG vs Hybrid on any email text.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils import (
    run_base_distilbert_spam, run_finetuned_distilbert_spam,
    run_rag_spam, run_hybrid_spam, get_demo_status,
)

st.set_page_config(page_title="Live Query - Spam Detection", page_icon="FT", layout="wide")

st.title("Live Query: Spam Detection")
st.markdown(
    "Enter an email and compare all four approaches side-by-side. "
    "All models use **DistilBERT 66M parameters** -- the difference is fine-tuning and retrieval."
)

with st.expander("Comparison Methodology"):
    st.markdown("""
    | Approach | Model | What changes |
    |----------|-------|-------------|
    | **Base DistilBERT** | distilbert-base-uncased (66M) | Nothing -- generic model, zero-shot cosine similarity |
    | **Fine-Tuned** | DistilBERT fine-tuned on spam (66M) | Trained on phishing/spam dataset |
    | **DistilBERT + RAG** | distilbert-base-uncased + KB | Retrieval of similar labeled emails, similarity voting |
    | **Hybrid** | Fine-Tuned + RAG | Fine-tuned prediction weighted with RAG neighbor vote |
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

# Sample emails
SAMPLES = [
    "Subject: Congratulations! You've won a $1,000,000 lottery prize! Click here to claim.",
    "Subject: Q4 Board Meeting Agenda\n\nHi team, attached is the agenda for our quarterly board meeting.",
    "Subject: URGENT: Your account has been compromised. Click below to verify your identity.",
    "Subject: Your Amazon order has shipped and is expected to arrive by Tuesday.",
    "Subject: Make $5000 per day from home! No experience needed. Act now!",
    "Subject: Re: Project timeline update\n\nThanks for the update. Can we sync tomorrow?",
    "Subject: Verify your PayPal account now or it will be permanently suspended.",
]

with st.form("spam_form"):
    text = st.text_area(
        "Email text:",
        value=SAMPLES[0],
        height=80,
    )
    st.caption("Try these examples:")
    for i, sample in enumerate(SAMPLES[1:], 1):
        st.caption(f"{i}. {sample[:80]}...")
    submitted = st.form_submit_button("Detect Spam", type="primary",
                                       use_container_width=True)

if submitted and text.strip():
    st.divider()

    col_base, col_ft, col_rag, col_hyb = st.columns(4)

    with col_base:
        st.subheader("Base DistilBERT")
        with st.spinner("Running..."):
            try:
                r = run_base_distilbert_spam(text)
                color = "red" if r.label == "spam" else "green"
                st.markdown(f"### :{color}[{r.label.upper()}]")
                st.metric("Confidence", f"{r.confidence:.1%}")
                st.caption(f"Latency: {r.latency_ms:.0f}ms")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_ft:
        st.subheader("Fine-Tuned")
        st.caption("Spam-trained")
        with st.spinner("Running..."):
            try:
                r = run_finetuned_distilbert_spam(text)
                color = "red" if r.label == "spam" else "green"
                st.markdown(f"### :{color}[{r.label.upper()}]")
                st.metric("Confidence", f"{r.confidence:.1%}")
                st.caption(f"Latency: {r.latency_ms:.0f}ms")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_rag:
        st.subheader("DistilBERT + RAG")
        with st.spinner("Running..."):
            try:
                r = run_rag_spam(text)
                color = "red" if r.label == "spam" else "green"
                st.markdown(f"### :{color}[{r.label.upper()}]")
                st.metric("Confidence", f"{r.confidence:.1%}")
                st.caption(f"Latency: {r.latency_ms:.0f}ms")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_hyb:
        st.subheader("Hybrid")
        st.caption("Fine-Tuned + RAG")
        with st.spinner("Running..."):
            try:
                r = run_hybrid_spam(text)
                color = "red" if r.label == "spam" else "green"
                st.markdown(f"### :{color}[{r.label.upper()}]")
                st.metric("Confidence", f"{r.confidence:.1%}")
                st.caption(f"Latency: {r.latency_ms:.0f}ms")
            except Exception as e:
                st.error(f"Error: {e}")
