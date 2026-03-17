"""
Live Query - Side-by-side streaming comparison
All three approaches run on the same query with streaming output.
Each model runs via Ollama (Mistral-7B) with real RAG retrieval.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils import (stream_finetuned, stream_rag, stream_base,
                        stream_hybrid, get_demo_status, LLM_MODEL,
                        rag_num_chunks)

st.set_page_config(page_title="Live Query - All Models", page_icon="FT", layout="wide")

st.title("Live Query: Side-by-Side Streaming")
st.markdown(
    "Enter a financial question and watch all three approaches generate "
    "responses in real time. Open each model in its own browser tab "
    "using the sidebar links."
)

# Methodology note
with st.expander("Comparison Methodology"):
    st.markdown(f"""
    **All approaches use the same base model: {LLM_MODEL} (~7B parameters)**

    | Approach | Model | What changes | Parameters |
    |----------|-------|-------------|------------|
    | **Base** | {LLM_MODEL} | Nothing -- raw model | ~7B |
    | **RAG** | {LLM_MODEL} + ChromaDB | Retrieved documents added to prompt | ~7B + 22M embedder |
    | **Fine-Tuned** | {LLM_MODEL} + expert prompt | Financial-expert system prompt | ~7B |
    | **Hybrid** | {LLM_MODEL} + expert prompt + RAG | Both system prompt and retrieval | ~7B + 22M embedder |

    **RAG knowledge base:** {rag_num_chunks()} chunks from 12 financial documents
    (earnings reports, SEC filings, risk analysis), embedded with all-MiniLM-L6-v2
    and stored in ChromaDB.

    **Note on fine-tuning:** In this live demo, the "fine-tuned" approach uses a
    domain-expert system prompt to approximate fine-tuned behavior. A truly fine-tuned
    model (e.g. FinQA-7B) would have its weights updated during training, achieving
    61.2% accuracy on FinQA vs ~15% for RAG on the same 7B architecture.
    See the Benchmark Results page for those numbers.
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


# Input form
with st.form("query_form"):
    question = st.text_input(
        "Question:",
        value="What was the percentage change in total revenue from 2022 to 2023?",
    )
    table = st.text_area(
        "Financial table (optional, markdown):",
        value=(
            "| Segment | 2023 | 2022 | 2021 |\n"
            "|---------|------|------|------|\n"
            "| Consumer Banking | $12,450 | $11,200 | $10,500 |\n"
            "| Commercial Banking | $8,320 | $7,890 | $7,200 |\n"
            "| Investment Banking | $5,180 | $6,240 | $5,800 |"
        ),
        height=140,
    )
    context = st.text_area(
        "Context (optional):",
        value=(
            "The decrease in Investment Banking segment revenue was primarily "
            "driven by lower trading volumes and reduced M&A advisory fees."
        ),
        height=80,
    )
    submitted = st.form_submit_button("Run All Three", type="primary",
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
