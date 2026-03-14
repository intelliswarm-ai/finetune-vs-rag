"""
Sentiment Analysis Demo Page - LIVE
Compare FinBERT (fine-tuned) vs RAG-based sentiment classification.
This is the core demo showing why fine-tuning outperforms RAG for specialized tasks.

FinBERT: A BERT model fine-tuned on 50,000+ financial sentences.
RAG: Retrieves similar labeled examples from a knowledge base, then classifies.
"""
import streamlit as st
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils import run_finbert, run_rag_sentiment, get_demo_status

st.set_page_config(page_title="Sentiment Analysis", page_icon="FT", layout="wide")

st.title("Fine-Tuned vs RAG: Sentiment Analysis")
st.markdown("""
**The core comparison:** FinBERT (fine-tuned on financial text) vs a RAG-based approach
(retrieve similar examples + LLM classification). See how fine-tuning delivers faster,
more accurate, and more confident results.
""")

# Sidebar
with st.sidebar:
    st.header("Model Status")
    status = get_demo_status()

    if status.get("finbert") == "live":
        st.success("FinBERT: Loaded (local)")
        st.caption("Real inference on CPU")
    else:
        st.warning("FinBERT: Simulated")
        st.caption("Install torch + transformers for live")

    if status.get("ollama") == "live":
        st.success("[LIVE] Ollama connected")
    else:
        st.error("[OFFLINE] Ollama not reachable")

    st.divider()
    st.markdown("""
    **What to observe:**
    1. **Speed**: FinBERT is 50-100x faster
    2. **Confidence**: FinBERT gives calibrated probabilities
    3. **Domain language**: FinBERT understands "headwinds", "exposure", etc.
    4. **RAG retrieval**: See which examples RAG pulls (and why they may mislead)
    """)

    st.divider()
    st.markdown("""
    **How this demonstrates the presentation topic:**

    - **Fine-Tuned (FinBERT)**: Learned financial language patterns during training.
      The word "headwinds" is embedded in its weights as negative.

    - **RAG approach**: Retrieves similar labeled sentences and tries to classify
      based on retrieved context. It doesn't *understand* - it *looks up*.
    """)

# Sample sentences with expected labels and explanations
SENTIMENT_SAMPLES = [
    {
        "text": "Net interest income grew 12% driven by higher rates and loan growth.",
        "expected": "positive",
        "explanation": "Clear growth language with specific metrics",
        "why_rag_struggles": "RAG likely gets this right - straightforward positive language",
    },
    {
        "text": "Management expects headwinds from deposit competition to persist.",
        "expected": "negative",
        "explanation": "'Headwinds' is financial jargon for challenges - a key test!",
        "why_rag_struggles": "RAG may miss 'headwinds' - it's domain-specific negative language",
    },
    {
        "text": "The company maintained its quarterly dividend of $0.50 per share.",
        "expected": "neutral",
        "explanation": "'Maintained' = no change, status quo. Not positive or negative.",
        "why_rag_struggles": "'Dividend' might retrieve positive examples, but 'maintained' is neutral",
    },
    {
        "text": "Credit costs increased significantly due to commercial real estate exposure.",
        "expected": "negative",
        "explanation": "Increased costs + exposure = negative financial outlook",
        "why_rag_struggles": "'Increased' alone could be positive; context matters",
    },
    {
        "text": "Strong demand in core markets contributed to double-digit revenue growth.",
        "expected": "positive",
        "explanation": "Strong demand + double-digit growth = clearly positive",
        "why_rag_struggles": "RAG should get this - clear positive indicators",
    },
    {
        "text": "Total assets remained relatively unchanged from the prior quarter.",
        "expected": "neutral",
        "explanation": "'Relatively unchanged' = no significant movement",
        "why_rag_struggles": "Neutral statements are hardest - RAG may default to positive or negative",
    },
    {
        "text": "The restructuring program resulted in $450M of one-time charges.",
        "expected": "negative",
        "explanation": "Restructuring charges are negative, even if strategic",
        "why_rag_struggles": "'Program' and specific dollar amounts might confuse retrieval",
    },
    {
        "text": "Operating efficiency improved with the cost-to-income ratio declining to 52%.",
        "expected": "positive",
        "explanation": "Improved efficiency + declining costs = positive",
        "why_rag_struggles": "'Declining' alone sounds negative, but declining costs is positive!",
    },
]

st.divider()

# ---------------------------------------------------------------------------
# Single text analysis
# ---------------------------------------------------------------------------
st.subheader("Analyze Single Text")

sample_texts = [s["text"] for s in SENTIMENT_SAMPLES]
selected_text = st.selectbox("Choose a sample sentence:", sample_texts)

custom_text = st.text_input("Or enter your own financial text:", "")
text_to_analyze = custom_text if custom_text.strip() else selected_text

# Show expected label for sample texts
sample_match = next((s for s in SENTIMENT_SAMPLES if s["text"] == text_to_analyze), None)
if sample_match:
    st.markdown(f"""
    **Expected:** `{sample_match['expected'].upper()}` - *{sample_match['explanation']}*

    **Why RAG may struggle:** *{sample_match['why_rag_struggles']}*
    """)

st.info(f"**Analyzing:** {text_to_analyze}")

if st.button("Run Fine-Tuned vs RAG Comparison", type="primary", use_container_width=True):
    st.divider()

    col1, col2 = st.columns(2)

    # -- FinBERT (fine-tuned) --
    with col1:
        st.markdown("### FinBERT (Fine-Tuned)")
        st.caption("BERT model fine-tuned on 50,000+ financial sentences")
        with st.spinner("Running FinBERT inference..."):
            finbert_result = run_finbert(text_to_analyze)

        # Color-coded result
        if finbert_result.label == "positive":
            st.success(f"**{finbert_result.label.upper()}** (confidence: {finbert_result.confidence:.0%})")
        elif finbert_result.label == "negative":
            st.error(f"**{finbert_result.label.upper()}** (confidence: {finbert_result.confidence:.0%})")
        else:
            st.warning(f"**{finbert_result.label.upper()}** (confidence: {finbert_result.confidence:.0%})")

        st.metric("Total Latency", f"{finbert_result.latency_ms:.1f} ms",
                   help="Single forward pass through the model")

        # Probability distribution
        st.markdown("**Calibrated Probability Distribution:**")
        for label, score in sorted(finbert_result.scores.items(),
                                    key=lambda x: x[1], reverse=True):
            st.progress(min(score, 1.0), text=f"{label}: {score:.1%}")

        st.markdown("""
        **How it works:**
        - Single forward pass through fine-tuned model
        - No retrieval, no external lookup
        - Domain knowledge is *in the weights*
        """)

        st.caption(f"Model: {finbert_result.model_name}")

    # -- RAG-based sentiment --
    with col2:
        st.markdown("### RAG-Based Sentiment")
        st.caption("Retrieve similar examples + LLM classification")
        with st.spinner("Retrieving similar examples & classifying..."):
            rag_result = run_rag_sentiment(text_to_analyze)

        if rag_result.label == "positive":
            st.success(f"**{rag_result.label.upper()}** (confidence: {rag_result.confidence:.0%})")
        elif rag_result.label == "negative":
            st.error(f"**{rag_result.label.upper()}** (confidence: {rag_result.confidence:.0%})")
        else:
            st.warning(f"**{rag_result.label.upper()}** (confidence: {rag_result.confidence:.0%})")

        st.metric("Total Latency", f"{rag_result.latency_ms:.1f} ms",
                   help=f"Retrieval: {rag_result.retrieval_ms:.0f}ms + Generation: {rag_result.generation_ms:.0f}ms")

        # Show latency breakdown
        st.markdown(f"""
        **Latency Breakdown:**
        - Embedding + Retrieval: `{rag_result.retrieval_ms:.0f} ms`
        - LLM Generation: `{rag_result.generation_ms:.0f} ms`
        """)

        # Show retrieved examples
        st.markdown("**Retrieved Similar Examples:**")
        for ex in rag_result.retrieved_examples:
            label_indicator = {"positive": "[+]", "negative": "[-]", "neutral": "[~]"}.get(ex["label"], "[?]")
            st.markdown(f"> {label_indicator} *\"{ex['text'][:80]}...\"* -> **{ex['label']}**")

        # Score distribution
        st.markdown("**Score Distribution:**")
        for label, score in sorted(rag_result.scores.items(),
                                    key=lambda x: x[1], reverse=True):
            st.progress(min(max(score, 0.0), 1.0), text=f"{label}: {score:.0%}")

        st.markdown("""
        **How it works:**
        1. Embed the input text
        2. Retrieve similar labeled examples
        3. LLM classifies based on retrieved context
        """)

        st.caption(f"Model: {rag_result.model_name}")

    # Comparison table
    st.divider()
    st.subheader("Side-by-Side Comparison")

    import pandas as pd

    # Calculate speed difference
    if rag_result.latency_ms > 0 and finbert_result.latency_ms > 0:
        speed_ratio = rag_result.latency_ms / max(finbert_result.latency_ms, 0.1)
    else:
        speed_ratio = 50

    comparison = {
        "Metric": [
            "Predicted Label",
            "Confidence Score",
            "Total Latency",
            "Calibrated Probabilities",
            "Needs Retrieval Step",
            "Understands Domain Jargon",
            "Consistent Across Runs",
        ],
        "FinBERT (Fine-Tuned)": [
            finbert_result.label.upper(),
            f"{finbert_result.confidence:.0%}",
            f"{finbert_result.latency_ms:.1f} ms",
            "Yes - trained probability distribution",
            "No - knowledge in model weights",
            "Yes - trained on financial text",
            "Yes - deterministic",
        ],
        "RAG-Based": [
            rag_result.label.upper(),
            f"{rag_result.confidence:.0%}",
            f"{rag_result.latency_ms:.1f} ms ({speed_ratio:.0f}x slower)",
            "No - rough vote/estimate",
            "Yes - retrieve + generate",
            "Depends on retrieved examples",
            "Variable - depends on retrieval",
        ],
    }
    st.table(pd.DataFrame(comparison))

    # Result analysis
    if sample_match:
        finbert_correct = finbert_result.label == sample_match["expected"]
        rag_correct = rag_result.label == sample_match["expected"]

        if finbert_correct and not rag_correct:
            st.error(f"""
            **FinBERT got it RIGHT, RAG got it WRONG!**

            This demonstrates exactly why fine-tuning matters:
            - FinBERT learned that "{text_to_analyze[:60]}..." is **{sample_match['expected']}**
              during training on financial text
            - RAG retrieved examples but couldn't correctly interpret the domain-specific language
            - **Fine-tuning teaches SKILLS (understanding), RAG provides INFORMATION (examples)**
            """)
        elif finbert_correct and rag_correct:
            st.success(f"""
            **Both got the correct label ({sample_match['expected'].upper()})**, but notice:
            - FinBERT confidence: **{finbert_result.confidence:.0%}** vs RAG: **{rag_result.confidence:.0%}**
            - FinBERT latency: **{finbert_result.latency_ms:.1f}ms** vs RAG: **{rag_result.latency_ms:.1f}ms** ({speed_ratio:.0f}x slower)
            - FinBERT provides **calibrated probability distribution**; RAG gives a rough estimate

            Even when RAG gets the right answer, fine-tuning is faster, more confident, and more reliable.
            """)
        elif not finbert_correct and rag_correct:
            st.info("Interesting! RAG got this one right. This can happen with straightforward cases "
                     "where the retrieved examples closely match.")
        else:
            st.warning("Both approaches struggled with this example.")

    st.success(f"""
    **Key Takeaway for this demo:**
    - FinBERT is **{speed_ratio:.0f}x faster** because there's no retrieval step
    - Fine-tuning gives **calibrated confidence scores** (probability distribution)
    - RAG depends on the **quality of retrieved examples** - if the wrong examples are retrieved, the answer is wrong
    - Fine-tuning **embeds domain knowledge in the model weights** - it doesn't just look things up
    """)

st.divider()

# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------
st.subheader("Batch Analysis: Fine-Tuned vs RAG Across All Samples")
st.markdown("Run both approaches on all 8 sample sentences and compare accuracy at scale.")

if st.button("Run Batch Comparison", type="primary", use_container_width=True):
    st.divider()

    results = []
    total_finbert_ms = 0
    total_rag_ms = 0
    progress_bar = st.progress(0, text="Running batch comparison...")

    for i, sample in enumerate(SENTIMENT_SAMPLES):
        finbert_r = run_finbert(sample["text"])
        rag_r = run_rag_sentiment(sample["text"])

        total_finbert_ms += finbert_r.latency_ms
        total_rag_ms += rag_r.latency_ms

        results.append({
            "Text": sample["text"][:60] + "...",
            "Expected": sample["expected"].upper(),
            "FinBERT": finbert_r.label.upper(),
            "FT Conf.": f"{finbert_r.confidence:.0%}",
            "FT OK": "Y" if finbert_r.label == sample["expected"] else "N",
            "RAG": rag_r.label.upper(),
            "RAG Conf.": f"{rag_r.confidence:.0%}",
            "RAG OK": "Y" if rag_r.label == sample["expected"] else "N",
        })

        progress_bar.progress((i + 1) / len(SENTIMENT_SAMPLES),
                              text=f"Analyzed {i + 1}/{len(SENTIMENT_SAMPLES)} texts...")

    import pandas as pd
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    # Summary metrics
    st.divider()
    finbert_correct = sum(1 for r in results if r["FT OK"] == "Y")
    rag_correct = sum(1 for r in results if r["RAG OK"] == "Y")
    total = len(results)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("FinBERT Accuracy",
                   f"{finbert_correct}/{total} ({finbert_correct/total:.0%})")
    with col2:
        st.metric("RAG Accuracy",
                   f"{rag_correct}/{total} ({rag_correct/total:.0%})")
    with col3:
        st.metric("FinBERT Total Time",
                   f"{total_finbert_ms:.0f} ms")
    with col4:
        st.metric("RAG Total Time",
                   f"{total_rag_ms:.0f} ms",
                   delta=f"{total_rag_ms/max(total_finbert_ms,1):.0f}x slower",
                   delta_color="inverse")

    st.success(f"""
    **Batch Results Summary:**
    - **FinBERT (fine-tuned):** {finbert_correct}/{total} correct ({finbert_correct/total:.0%}), total time {total_finbert_ms:.0f}ms
    - **RAG-based:** {rag_correct}/{total} correct ({rag_correct/total:.0%}), total time {total_rag_ms:.0f}ms
    - **Speed advantage:** FinBERT is {total_rag_ms/max(total_finbert_ms,1):.0f}x faster overall

    **Why?** Fine-tuning taught FinBERT the *language of finance*. Words like "headwinds",
    "exposure", and "restructuring" have specific negative connotations that the model learned
    during training. RAG can only retrieve similar examples - it doesn't truly *understand*
    the domain vocabulary.
    """)

st.divider()
st.caption("Live comparison: FinBERT (fine-tuned) vs RAG-based sentiment analysis")
