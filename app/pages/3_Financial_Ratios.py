"""
Financial Ratios Demo Page
Calculate and compare financial ratios using all three approaches.
Uses live model calls when available.
"""
import streamlit as st
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils import call_base_model, call_rag_model, call_finetuned_model, retrieve_documents

st.set_page_config(page_title="Financial Ratios", page_icon="FT", layout="wide")

st.title("Financial Ratio Analysis")
st.markdown("Calculate financial ratios with step-by-step reasoning. Compare fine-tuned vs RAG vs base model.")

# Sample ratio scenarios
RATIO_CASES = [
    {
        "name": "Debt-to-Equity Analysis",
        "table": """| Item | 2023 | 2022 |
|------|------|------|
| Total Debt | $52,300M | $48,900M |
| Shareholders' Equity | $47,400M | $43,800M |""",
        "question": "Calculate D/E ratio for both years. Is leverage improving?",
        "expected": "2023 D/E: 1.10, 2022 D/E: 1.12. Leverage is improving (decreasing).",
    },
    {
        "name": "Return on Equity (ROE)",
        "table": """| Item | 2023 | 2022 |
|------|------|------|
| Net Income | $8,200M | $7,500M |
| Avg Shareholders' Equity | $45,600M | $42,100M |""",
        "question": "Calculate ROE for 2023 and compare to 2022.",
        "expected": "2023 ROE: 18.0%, 2022 ROE: 17.8%. ROE improved slightly.",
    },
    {
        "name": "Current Ratio",
        "table": """| Item | Q4 2023 | Q3 2023 |
|------|---------|---------|
| Current Assets | $125,400M | $118,200M |
| Current Liabilities | $89,500M | $92,100M |""",
        "question": "Calculate the current ratio for both quarters. Is liquidity improving?",
        "expected": "Q4 2023: 1.40, Q3 2023: 1.28. Liquidity is improving.",
    },
]

# Case selector
selected = st.selectbox("Select ratio analysis:", [c["name"] for c in RATIO_CASES])
case = next(c for c in RATIO_CASES if c["name"] == selected)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Data")
    st.markdown(case["table"])
with col2:
    st.markdown("### Question")
    st.info(case["question"])
    st.success(f"**Expected:** {case['expected']}")

if st.button("Calculate Ratio", type="primary", use_container_width=True):
    st.divider()

    col_ft, col_rag, col_base = st.columns(3)

    with col_ft:
        st.markdown("### Fine-Tuned")
        with st.spinner("Computing..."):
            ft_r = call_finetuned_model(case["question"], case["table"])
        if ft_r.reasoning_steps:
            st.code(ft_r.reasoning_steps, language=None)
        st.markdown(f"**Answer:** {ft_r.answer}")
        st.metric("Latency", f"{ft_r.latency_ms:.0f} ms")

    with col_rag:
        st.markdown("### RAG")
        with st.spinner("Retrieving & generating..."):
            docs, _ = retrieve_documents(case["question"])
            rag_r = call_rag_model(case["question"], case["table"], retrieved_docs=docs)
        if rag_r.retrieved_context:
            for doc in rag_r.retrieved_context[:1]:
                st.markdown(f"> *{doc[:100]}...*")
        st.markdown(f"**Answer:** {rag_r.answer}")
        st.metric("Latency", f"{rag_r.latency_ms:.0f} ms")

    with col_base:
        st.markdown("### Base LLM")
        with st.spinner("Querying..."):
            base_r = call_base_model(case["question"], case["table"])
        st.markdown(f"**Answer:** {base_r.answer}")
        st.metric("Latency", f"{base_r.latency_ms:.0f} ms")

    st.divider()
    st.success("""
    **Key Insight:** The fine-tuned model provides precise, step-by-step calculations with exact values.
    RAG retrieves the formula definition but the base model may not apply it accurately.
    """)

st.divider()
st.caption("Financial ratio calculations with real-time comparison")
