"""
Live Query - Financial Ratios
Compare Base Llama2 vs FinQA-7B vs RAG vs Hybrid on financial ratio calculations.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils import (stream_finetuned, stream_rag, stream_base,
                        stream_hybrid, get_demo_status, LLM_MODEL,
                        rag_num_chunks)

st.set_page_config(page_title="Live Query - Financial Ratios", page_icon="FT", layout="wide")

st.title("Live Query: Financial Ratios")
st.markdown(
    "Enter a financial ratio question with table data and compare all four approaches. "
    "All models use **Llama2-7B** -- the difference is fine-tuning and retrieval."
)

with st.expander("Comparison Methodology"):
    st.markdown(f"""
    **All approaches use the same base model: {LLM_MODEL} (~7B parameters)**

    | Approach | Model | What changes | Parameters |
    |----------|-------|-------------|------------|
    | **Base** | {LLM_MODEL} | Nothing -- raw model | ~7B |
    | **RAG** | {LLM_MODEL} + ChromaDB | Retrieved documents added to prompt | ~7B + 22M embedder |
    | **Fine-Tuned** | {LLM_MODEL} + expert prompt | Financial-expert system prompt | ~7B |
    | **Hybrid** | {LLM_MODEL} + expert prompt + RAG | Both system prompt and retrieval | ~7B + 22M embedder |

    **RAG knowledge base:** {rag_num_chunks()} chunks from 12 financial documents,
    embedded with all-MiniLM-L6-v2 and stored in ChromaDB.
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

# Sample questions
SAMPLES = {
    "DuPont ROE": {
        "question": "Decompose the Return on Equity using the 3-component DuPont formula: ROE = (Net Income/Revenue) x (Revenue/Total Assets) x (Total Assets/Equity). What is the ROE percentage?",
        "table": "| Item | 2023 |\n|---|---|\n| Net Income | $8,200M |\n| Revenue | $56,300M |\n| Total Assets (avg) | $187,600M |\n| Shareholders' Equity (avg) | $45,600M |",
        "context": "Analysts use the DuPont framework to decompose return on equity into profitability, efficiency, and leverage components.",
    },
    "CAGR": {
        "question": "Calculate the 3-year Compound Annual Growth Rate (CAGR) of revenue from 2020 to 2023. CAGR = (End/Start)^(1/n) - 1.",
        "table": "| Item | 2023 | 2020 |\n|---|---|---|\n| Revenue | $56,300M | $42,100M |",
        "context": "The company achieved consistent revenue growth over the three-year period from 2020 to 2023.",
    },
    "Debt-to-Equity": {
        "question": "Calculate the debt-to-equity ratio by first summing all debt components and all equity components, then dividing total debt by total equity.",
        "table": "| Item | 2023 |\n|---|---|\n| Short-term Debt | $12,400M |\n| Long-term Debt | $55,400M |\n| Common Stock | $18,200M |\n| Additional Paid-in Capital | $9,800M |\n| Retained Earnings | $19,400M |",
        "context": "The company's capital structure includes multiple debt and equity components.",
    },
}

selected = st.selectbox("Choose a sample question:", list(SAMPLES.keys()))
sample = SAMPLES[selected]

with st.form("ratio_form"):
    question = st.text_input("Question:", value=sample["question"])
    table = st.text_area("Financial table (markdown):", value=sample["table"], height=140)
    context = st.text_area("Context:", value=sample["context"], height=80)
    submitted = st.form_submit_button("Run All Four", type="primary",
                                       use_container_width=True)

if submitted:
    st.divider()

    col_ft, col_rag, col_base = st.columns(3)

    with col_ft:
        st.subheader("Fine-Tuned")
        st.caption(f"{LLM_MODEL} + expert system prompt")
        st.write_stream(stream_finetuned(question, table, context))

    with col_rag:
        st.subheader("RAG")
        st.caption(f"{LLM_MODEL} + ChromaDB retrieval")
        st.write_stream(stream_rag(question, table, context))

    with col_base:
        st.subheader("Base LLM")
        st.caption(f"{LLM_MODEL} (no RAG, no prompt)")
        st.write_stream(stream_base(question, table, context))

    st.divider()
    st.subheader("Hybrid (Fine-Tuned + RAG)")
    st.caption(f"{LLM_MODEL} + expert system prompt + ChromaDB retrieval")
    st.write_stream(stream_hybrid(question, table, context))
