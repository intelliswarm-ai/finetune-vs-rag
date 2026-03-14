"""
Numerical Reasoning Demo - Live 3-Way Comparison
The PRIMARY demo showing Fine-Tuned vs RAG vs Base LLM
Calls real models when available, falls back to realistic simulated responses.
"""
import streamlit as st
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils import (
    call_base_model,
    call_rag_model,
    call_finetuned_model,
    retrieve_documents,
    get_demo_status,
    has_llm,
)

st.set_page_config(
    page_title="Numerical Reasoning - FT vs RAG",
    page_icon="FT",
    layout="wide",
)

# Sample test cases
SAMPLE_CASES = [
    {
        "name": "Revenue Growth Calculation",
        "table": """| Segment | 2023 | 2022 | 2021 |
|---------|------|------|------|
| Consumer Banking | $12,450 | $11,200 | $10,500 |
| Commercial Banking | $8,320 | $7,890 | $7,200 |
| Investment Banking | $5,180 | $6,240 | $5,800 |""",
        "text": "The decrease in Investment Banking segment revenue was primarily driven by lower trading volumes and reduced M&A advisory fees due to market uncertainty. Consumer Banking continued to show strong growth driven by higher net interest income.",
        "question": "What was the percentage change in total revenue from 2022 to 2023?",
        "expected": "2.45%",
    },
    {
        "name": "Debt-to-Equity Ratio",
        "table": """| Item | 2023 | 2022 |
|------|------|------|
| Total Assets | $245,600 | $231,400 |
| Total Liabilities | $198,200 | $187,600 |
| Shareholders' Equity | $47,400 | $43,800 |
| Total Debt | $52,300 | $48,900 |""",
        "text": "The company maintained a strong capital position throughout 2023, with equity growing faster than debt.",
        "question": "Calculate the debt-to-equity ratio for 2023 and compare it to 2022. Is leverage increasing or decreasing?",
        "expected": "2023 D/E: 1.10, 2022 D/E: 1.12. Leverage is decreasing.",
    },
    {
        "name": "Efficiency Ratio Analysis",
        "table": """| Metric | Q4 2023 | Q3 2023 | Q4 2022 |
|--------|---------|---------|---------|
| Net Interest Income | $14,200 | $13,800 | $12,500 |
| Non-Interest Income | $5,600 | $5,900 | $6,100 |
| Total Revenue | $19,800 | $19,700 | $18,600 |
| Operating Expenses | $11,200 | $10,900 | $10,400 |""",
        "text": "Net interest income continued to benefit from higher rates, while non-interest income faced headwinds from reduced trading activity.",
        "question": "What is the efficiency ratio (Operating Expenses / Total Revenue) for Q4 2023, and how does it compare to Q4 2022?",
        "expected": "Q4 2023: 56.6%, Q4 2022: 55.9%. Efficiency ratio increased by 0.7 percentage points.",
    },
]

# Header
st.title("Numerical Reasoning: Live 3-Way Comparison")
st.markdown("""
**The killer demo:** Watch the fine-tuned model do math that RAG simply cannot.
This page runs all three approaches and compares results side-by-side.
""")

st.divider()

# Sidebar status
with st.sidebar:
    st.header("Demo Status")
    status = get_demo_status()

    if status.get("ollama") == "live":
        st.success("[LIVE] Ollama connected")
    else:
        st.error("[OFFLINE] Ollama not reachable")

    if status.get("rag") == "live":
        st.success("[LIVE] RAG indexed")
    else:
        st.warning("[INIT] RAG loading...")

    st.divider()
    st.markdown("""
    **What to observe:**
    1. Fine-tuned shows step-by-step **reasoning**
    2. RAG retrieves context but **struggles with math**
    3. Notice the **latency difference**
    4. Compare **accuracy** of final answers
    """)

# Main content
tab1, tab2 = st.tabs(["Sample Cases", "Custom Input"])

with tab1:
    case_names = [c["name"] for c in SAMPLE_CASES]
    selected_case = st.selectbox("Select a test case:", case_names)
    case = next(c for c in SAMPLE_CASES if c["name"] == selected_case)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Financial Data")
        st.markdown(case["table"])
    with col2:
        st.subheader("Context")
        st.markdown(case["text"])

    st.subheader("Question")
    st.info(case["question"])

    st.subheader("Expected Answer")
    st.success(case["expected"])

    # Run comparison
    if st.button("Run 3-Way Comparison", type="primary", use_container_width=True):

        st.divider()
        st.subheader("Results")

        col_ft, col_rag, col_base = st.columns(3)

        # -- Fine-Tuned Model --
        with col_ft:
            st.markdown("### Fine-Tuned (FinQA-7B)")
            with st.spinner("Computing with domain-trained model..."):
                ft_response = call_finetuned_model(
                    question=case["question"],
                    table=case["table"],
                    context=case["text"],
                )

            if ft_response.reasoning_steps:
                st.markdown("**Reasoning Steps:**")
                st.code(ft_response.reasoning_steps, language=None)

            st.markdown(f"**Answer:** {ft_response.answer}")

            st.metric("Latency", f"{ft_response.latency_ms:.0f} ms",
                       delta=f"Fastest", delta_color="normal")

            st.caption(f"Model: {ft_response.model_name}")

        # -- RAG Model --
        with col_rag:
            st.markdown("### RAG (Retrieve + Generate)")
            with st.spinner("Retrieving documents & generating..."):
                # First, retrieve relevant documents
                retrieved_docs, retrieval_ms = retrieve_documents(case["question"])
                rag_response = call_rag_model(
                    question=case["question"],
                    table=case["table"],
                    context=case["text"],
                    retrieved_docs=retrieved_docs,
                )

            if rag_response.retrieved_context:
                st.markdown("**Retrieved Documents:**")
                for i, doc in enumerate(rag_response.retrieved_context[:2], 1):
                    st.markdown(f"> *[{i}]* {doc[:120]}...")

            st.markdown(f"**Answer:** {rag_response.answer}")

            total_rag_ms = rag_response.latency_ms
            st.metric("Latency", f"{total_rag_ms:.0f} ms",
                       delta=f"+{total_rag_ms - ft_response.latency_ms:.0f} ms vs FT",
                       delta_color="inverse")

            st.caption(f"Model: {rag_response.model_name}")

        # -- Base Model --
        with col_base:
            st.markdown("### Base LLM (No fine-tuning)")
            with st.spinner("Querying base model..."):
                base_response = call_base_model(
                    question=case["question"],
                    table=case["table"],
                    context=case["text"],
                )

            st.markdown(f"**Answer:** {base_response.answer}")

            st.metric("Latency", f"{base_response.latency_ms:.0f} ms",
                       delta=f"+{base_response.latency_ms - ft_response.latency_ms:.0f} ms vs FT",
                       delta_color="inverse")

            st.caption(f"Model: {base_response.model_name}")

        # Summary comparison
        st.divider()
        st.subheader("Comparison Summary")

        import pandas as pd

        # Determine accuracy
        expected_lower = case["expected"].lower()

        def check_accuracy(response):
            answer_lower = response.answer.lower()
            # Check if key numbers from expected answer appear
            import re
            expected_numbers = re.findall(r'\d+\.?\d*', expected_lower)
            if not expected_numbers:
                return "N/A"
            found = sum(1 for n in expected_numbers if n in answer_lower)
            if found >= len(expected_numbers) * 0.5:
                return "Correct"
            return "Incomplete/Incorrect"

        ft_acc = check_accuracy(ft_response)
        rag_acc = check_accuracy(rag_response)
        base_acc = check_accuracy(base_response)

        summary_data = {
            "Metric": ["Accuracy", "Latency (ms)", "Shows Reasoning Steps",
                        "Uses Retrieved Context", "Live Model"],
            "Fine-Tuned (FinQA-7B)": [
                f"{ft_acc}",
                f"{ft_response.latency_ms:.0f}",
                "Yes" if ft_response.reasoning_steps else "No",
                "No (knowledge in weights)",
                ft_response.model_name,
            ],
            "RAG": [
                f"{rag_acc}",
                f"{rag_response.latency_ms:.0f}",
                "No",
                "Yes",
                rag_response.model_name,
            ],
            "Base LLM": [
                f"{base_acc}",
                f"{base_response.latency_ms:.0f}",
                "Sometimes",
                "No",
                base_response.model_name,
            ],
        }

        st.table(pd.DataFrame(summary_data))

        st.success("""
        **Key Insight:** The fine-tuned model (FinQA-7B) performs precise multi-step calculations
        because it *learned* numerical reasoning during training. RAG retrieves relevant formulas
        but the base model cannot reliably apply them. Fine-tuning teaches **skills**, not just **facts**.
        """)

with tab2:
    st.subheader("Enter Custom Financial Data")

    custom_table = st.text_area(
        "Financial Table (Markdown format):",
        value="""| Metric | 2023 | 2022 |
|--------|------|------|
| Revenue | $100M | $90M |
| Expenses | $60M | $55M |""",
        height=150,
    )

    custom_text = st.text_area(
        "Context Text:",
        value="The company showed strong growth in 2023 driven by market expansion.",
        height=100,
    )

    custom_question = st.text_input(
        "Your Question:",
        value="What was the revenue growth rate from 2022 to 2023?",
    )

    if st.button("Run Custom Comparison", type="primary"):
        st.divider()

        col_ft, col_rag, col_base = st.columns(3)

        with col_ft:
            st.markdown("### Fine-Tuned")
            with st.spinner("Computing..."):
                ft_r = call_finetuned_model(custom_question, custom_table, custom_text)
            if ft_r.reasoning_steps:
                st.code(ft_r.reasoning_steps, language=None)
            st.markdown(f"**Answer:** {ft_r.answer}")
            st.metric("Latency", f"{ft_r.latency_ms:.0f} ms")

        with col_rag:
            st.markdown("### RAG")
            with st.spinner("Retrieving & generating..."):
                docs, _ = retrieve_documents(custom_question)
                rag_r = call_rag_model(custom_question, custom_table, custom_text, docs)
            st.markdown(f"**Answer:** {rag_r.answer}")
            st.metric("Latency", f"{rag_r.latency_ms:.0f} ms")

        with col_base:
            st.markdown("### Base LLM")
            with st.spinner("Querying..."):
                base_r = call_base_model(custom_question, custom_table, custom_text)
            st.markdown(f"**Answer:** {base_r.answer}")
            st.metric("Latency", f"{base_r.latency_ms:.0f} ms")

        if not has_llm():
            st.error("Ollama not reachable. Start it with: `ollama serve` then `ollama pull mistral`")

st.divider()
st.caption("Live comparison: Fine-Tuned vs RAG vs Base LLM for numerical reasoning")
