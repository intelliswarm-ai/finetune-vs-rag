"""
Spam Detection Demo Page - LIVE
Compare Fine-tuned DistilBERT (spam-trained) vs RAG-based spam classification.
Demonstrates fine-tuning vs RAG for binary classification on email spam detection.

Fine-tuned: DistilBERT model trained on phishing/spam dataset.
RAG: Retrieves similar labeled examples from a knowledge base, then classifies.
"""
import streamlit as st
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils import (
    run_finetuned_distilbert_spam, run_rag_spam, get_demo_status,
)

st.set_page_config(page_title="Spam Detection", page_icon="FT", layout="wide")

st.title("Fine-Tuned vs RAG: Spam Detection")
st.markdown("""
**The core comparison:** Fine-tuned DistilBERT (trained on phishing/spam data) vs a RAG-based approach
(retrieve similar labeled examples + similarity voting). See how fine-tuning delivers faster,
more accurate, and more confident results on spam detection.
""")

# Sidebar
with st.sidebar:
    st.header("Model Status")
    status = get_demo_status()

    if status.get("distilbert-ft-spam") == "live":
        st.success("Fine-tuned DistilBERT: Loaded (local)")
        st.caption("Real inference on CPU")
    else:
        st.warning("Fine-tuned DistilBERT: Not loaded")
        st.caption("Run: python app/download_spam_model.py")

    if status.get("distilbert-base-spam") == "live":
        st.success("Base DistilBERT: Loaded (local)")
    else:
        st.warning("Base DistilBERT: Not loaded")

    st.divider()
    st.markdown("""
    **What to observe:**
    1. **Speed**: Fine-tuned model runs in a single pass
    2. **Confidence**: Fine-tuned gives calibrated probabilities
    3. **Phishing detection**: Fine-tuned recognizes urgency, suspicious links, etc.
    4. **RAG retrieval**: See which examples RAG pulls (and why they may mislead)
    """)

    st.divider()
    st.markdown("""
    **How this demonstrates the presentation topic:**

    - **Fine-Tuned (DistilBERT)**: Learned phishing/spam patterns during training.
      Words like "urgent", "verify", "click here" are embedded as spam indicators.

    - **RAG approach**: Retrieves similar labeled emails and classifies
      based on neighbor voting. It doesn't *understand* -- it *looks up*.
    """)

# Sample emails
SPAM_SAMPLES = [
    {
        "text": "Subject: Congratulations! You've won a $1,000,000 lottery prize! Click here to claim your reward now.",
        "expected": "spam",
        "explanation": "Classic lottery scam with urgency and prize claim language",
        "why_rag_struggles": "RAG should get this -- clear spam indicators",
    },
    {
        "text": "Subject: Q4 Board Meeting Agenda\n\nHi team, attached is the agenda for our quarterly board meeting scheduled for March 25th.",
        "expected": "ham",
        "explanation": "Normal business email about a meeting",
        "why_rag_struggles": "RAG should get this -- straightforward business language",
    },
    {
        "text": "Subject: URGENT: Your account has been compromised\n\nDear valued customer, we detected suspicious activity. Click the link below to verify your identity immediately.",
        "expected": "spam",
        "explanation": "Phishing attempt with urgency, suspicious activity claim, and verification request",
        "why_rag_struggles": "Fine-tuned model knows this pattern from training; RAG must find similar examples",
    },
    {
        "text": "Subject: Your prescription is ready for pickup\n\nHi, this is a reminder that your prescription is ready for pickup at Walgreens on Main St.",
        "expected": "ham",
        "explanation": "Legitimate notification from a pharmacy",
        "why_rag_struggles": "The word 'prescription' might trigger spam detection if RAG retrieves medication spam examples",
    },
    {
        "text": "Subject: Verify your PayPal account now\n\nWe noticed unusual login activity. Please confirm your account details within 24 hours to avoid suspension.",
        "expected": "spam",
        "explanation": "PayPal phishing -- urgency + account verification + suspension threat",
        "why_rag_struggles": "Fine-tuned model learned this pattern; RAG needs exact matches",
    },
    {
        "text": "Subject: Re: Project timeline update\n\nThanks for the update. I've adjusted the milestones in Jira. Can we sync tomorrow at 2pm?",
        "expected": "ham",
        "explanation": "Normal project management email",
        "why_rag_struggles": "RAG should get this -- clear business context",
    },
    {
        "text": "Subject: Make $5000 per day from home!\n\nDiscover the secret system that millionaires use. No experience needed. Limited spots available!",
        "expected": "spam",
        "explanation": "Get-rich-quick scam with exaggerated claims",
        "why_rag_struggles": "Both should catch this -- obvious spam indicators",
    },
    {
        "text": "Subject: Your Amazon order has shipped\n\nYour order #112-4567890-1234567 has shipped and is expected to arrive by Tuesday.",
        "expected": "ham",
        "explanation": "Legitimate shipping notification with order number",
        "why_rag_struggles": "Could be confused with phishing if RAG retrieves fake shipping spam",
    },
]

st.divider()

# ---------------------------------------------------------------------------
# Single text analysis
# ---------------------------------------------------------------------------
st.subheader("Analyze Single Email")

sample_texts = [s["text"] for s in SPAM_SAMPLES]
selected_text = st.selectbox("Choose a sample email:", sample_texts)

custom_text = st.text_area("Or enter your own email text:", "", height=80)
text_to_analyze = custom_text if custom_text.strip() else selected_text

# Show expected label
sample_match = next((s for s in SPAM_SAMPLES if s["text"] == text_to_analyze), None)
if sample_match:
    st.markdown(f"""
    **Expected:** `{sample_match['expected'].upper()}` -- *{sample_match['explanation']}*

    **Why RAG may struggle:** *{sample_match['why_rag_struggles']}*
    """)

st.info(f"**Analyzing:** {text_to_analyze[:100]}...")

if st.button("Run Fine-Tuned vs RAG Comparison", type="primary", use_container_width=True):
    st.divider()

    col1, col2 = st.columns(2)

    # -- Fine-tuned DistilBERT --
    with col1:
        st.markdown("### Fine-Tuned DistilBERT")
        st.caption("DistilBERT fine-tuned on phishing/spam dataset")
        with st.spinner("Running fine-tuned inference..."):
            try:
                ft_result = run_finetuned_distilbert_spam(text_to_analyze)

                if ft_result.label == "spam":
                    st.error(f"**{ft_result.label.upper()}** (confidence: {ft_result.confidence:.0%})")
                else:
                    st.success(f"**{ft_result.label.upper()}** (confidence: {ft_result.confidence:.0%})")

                st.metric("Total Latency", f"{ft_result.latency_ms:.1f} ms",
                          help="Single forward pass through the model")

                st.markdown("**Probability Distribution:**")
                for label, score in sorted(ft_result.scores.items(),
                                           key=lambda x: x[1], reverse=True):
                    st.progress(min(score, 1.0), text=f"{label}: {score:.1%}")

                st.markdown("""
                **How it works:**
                - Single forward pass through fine-tuned model
                - No retrieval, no external lookup
                - Spam patterns are *in the weights*
                """)

                st.caption(f"Model: {ft_result.model_name}")
            except Exception as e:
                st.error(f"Error: {e}")

    # -- RAG-based spam detection --
    with col2:
        st.markdown("### RAG-Based Spam Detection")
        st.caption("Retrieve similar examples + similarity voting")
        with st.spinner("Retrieving similar examples & classifying..."):
            try:
                rag_result = run_rag_spam(text_to_analyze)

                if rag_result.label == "spam":
                    st.error(f"**{rag_result.label.upper()}** (confidence: {rag_result.confidence:.0%})")
                else:
                    st.success(f"**{rag_result.label.upper()}** (confidence: {rag_result.confidence:.0%})")

                st.metric("Total Latency", f"{rag_result.latency_ms:.1f} ms",
                          help=f"Retrieval: {rag_result.retrieval_ms:.0f}ms + Voting: {rag_result.generation_ms:.0f}ms")

                st.markdown(f"""
                **Latency Breakdown:**
                - Embedding + Retrieval: `{rag_result.retrieval_ms:.0f} ms`
                - Voting: `{rag_result.generation_ms:.0f} ms`
                """)

                st.markdown("**Retrieved Similar Examples:**")
                for ex in rag_result.retrieved_examples:
                    label_icon = "[SPAM]" if ex["label"] == "spam" else "[HAM]"
                    st.markdown(f"> {label_icon} *\"{ex['text'][:80]}...\"* -> **{ex['label']}**")

                st.markdown("**Score Distribution:**")
                for label, score in sorted(rag_result.scores.items(),
                                           key=lambda x: x[1], reverse=True):
                    st.progress(min(max(score, 0.0), 1.0), text=f"{label}: {score:.0%}")

                st.markdown("""
                **How it works:**
                1. Embed the input email
                2. Retrieve similar labeled examples
                3. Similarity-weighted voting determines label
                """)

                st.caption(f"Model: {rag_result.model_name}")
            except Exception as e:
                st.error(f"Error: {e}")

    # Comparison
    st.divider()
    st.subheader("Side-by-Side Comparison")

    import pandas as pd

    try:
        speed_ratio = rag_result.latency_ms / max(ft_result.latency_ms, 0.1)

        comparison = {
            "Metric": [
                "Predicted Label",
                "Confidence Score",
                "Total Latency",
                "Needs Retrieval Step",
                "Understands Phishing Patterns",
                "Consistent Across Runs",
            ],
            "Fine-Tuned DistilBERT": [
                ft_result.label.upper(),
                f"{ft_result.confidence:.0%}",
                f"{ft_result.latency_ms:.1f} ms",
                "No - knowledge in model weights",
                "Yes - trained on phishing data",
                "Yes - deterministic",
            ],
            "RAG-Based": [
                rag_result.label.upper(),
                f"{rag_result.confidence:.0%}",
                f"{rag_result.latency_ms:.1f} ms ({speed_ratio:.0f}x slower)",
                "Yes - retrieve + vote",
                "Depends on retrieved examples",
                "Variable - depends on retrieval",
            ],
        }
        st.table(pd.DataFrame(comparison))

        if sample_match:
            ft_correct = ft_result.label == sample_match["expected"]
            rag_correct = rag_result.label == sample_match["expected"]

            if ft_correct and not rag_correct:
                st.error(f"""
                **Fine-Tuned got it RIGHT, RAG got it WRONG!**

                Fine-tuning learned spam patterns during training. The model recognized
                phishing indicators like urgency, suspicious links, and verification requests
                from its training data.
                """)
            elif ft_correct and rag_correct:
                st.success(f"""
                **Both correctly identified this as {sample_match['expected'].upper()}**, but notice:
                - Fine-tuned confidence: **{ft_result.confidence:.0%}** vs RAG: **{rag_result.confidence:.0%}**
                - Fine-tuned latency: **{ft_result.latency_ms:.1f}ms** vs RAG: **{rag_result.latency_ms:.1f}ms**

                Even when RAG gets the right answer, fine-tuning is faster and more confident.
                """)
            elif not ft_correct and rag_correct:
                st.info("RAG got this one right. This can happen when retrieved examples closely match.")
            else:
                st.warning("Both approaches struggled with this example.")
    except Exception:
        pass

st.divider()

# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------
st.subheader("Batch Analysis: Fine-Tuned vs RAG Across All Samples")
st.markdown("Run both approaches on all 8 sample emails and compare accuracy at scale.")

if st.button("Run Batch Comparison", type="primary", use_container_width=True):
    st.divider()

    results = []
    total_ft_ms = 0
    total_rag_ms = 0
    progress_bar = st.progress(0, text="Running batch comparison...")

    for i, sample in enumerate(SPAM_SAMPLES):
        ft_r = run_finetuned_distilbert_spam(sample["text"])
        rag_r = run_rag_spam(sample["text"])

        total_ft_ms += ft_r.latency_ms
        total_rag_ms += rag_r.latency_ms

        results.append({
            "Email": sample["text"][:60] + "...",
            "Expected": sample["expected"].upper(),
            "Fine-Tuned": ft_r.label.upper(),
            "FT Conf.": f"{ft_r.confidence:.0%}",
            "FT OK": "Y" if ft_r.label == sample["expected"] else "N",
            "RAG": rag_r.label.upper(),
            "RAG Conf.": f"{rag_r.confidence:.0%}",
            "RAG OK": "Y" if rag_r.label == sample["expected"] else "N",
        })

        progress_bar.progress((i + 1) / len(SPAM_SAMPLES),
                              text=f"Analyzed {i + 1}/{len(SPAM_SAMPLES)} emails...")

    import pandas as pd
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    st.divider()
    ft_correct = sum(1 for r in results if r["FT OK"] == "Y")
    rag_correct = sum(1 for r in results if r["RAG OK"] == "Y")
    total = len(results)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fine-Tuned Accuracy",
                   f"{ft_correct}/{total} ({ft_correct/total:.0%})")
    with col2:
        st.metric("RAG Accuracy",
                   f"{rag_correct}/{total} ({rag_correct/total:.0%})")
    with col3:
        st.metric("Fine-Tuned Total Time",
                   f"{total_ft_ms:.0f} ms")
    with col4:
        st.metric("RAG Total Time",
                   f"{total_rag_ms:.0f} ms",
                   delta=f"{total_rag_ms/max(total_ft_ms,1):.0f}x slower",
                   delta_color="inverse")

    st.success(f"""
    **Batch Results Summary:**
    - **Fine-Tuned DistilBERT:** {ft_correct}/{total} correct ({ft_correct/total:.0%}), total time {total_ft_ms:.0f}ms
    - **RAG-based:** {rag_correct}/{total} correct ({rag_correct/total:.0%}), total time {total_rag_ms:.0f}ms

    **Why?** Fine-tuning taught the model phishing/spam patterns -- urgency language,
    suspicious URLs, verification requests, and prize scams are embedded in its weights.
    RAG can only retrieve similar examples -- it doesn't truly *understand* spam patterns.
    """)

st.divider()
st.caption("Live comparison: Fine-tuned DistilBERT vs RAG-based spam detection")
