"""
Hybrid Approach - Standalone query page
RAG retrieval + fine-tuned-style generation via Ollama.
URL: http://localhost:8501/Hybrid
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils import stream_hybrid, get_demo_status, LLM_MODEL, rag_num_chunks

st.set_page_config(page_title="Hybrid Model", page_icon="FT", layout="wide")

st.title("Hybrid Model Query")
st.markdown(
    "Best of both worlds: retrieve relevant documents (RAG) then "
    f"generate with **{LLM_MODEL}** (7B params) using a financial-expert "
    "system prompt."
)

with st.sidebar:
    st.header("Model Card")
    status = get_demo_status()
    if status.get("ollama") == "live":
        st.success(f"[LIVE] {LLM_MODEL}")
    else:
        st.error("[OFFLINE] Ollama not reachable")
    if status.get("rag") == "live":
        st.success(f"[LIVE] RAG: {rag_num_chunks()} chunks indexed")
    else:
        st.warning("[INIT] RAG initializing...")

    st.markdown(f"""
    **Generator:** {LLM_MODEL} (~7B params)
    **+ System prompt:** Financial expert
    **+ Retrieval:** ChromaDB top-3 chunks
    **Documents:** 12 financial reports

    ---

    **How it works:**
    1. Retrieve supporting docs (RAG)
    2. Feed docs + data to {LLM_MODEL}
       with financial-expert system prompt
    3. Model generates with domain knowledge
       AND fresh document context

    **Benchmark (FinQA-7B + RAG):**
    - Highest accuracy: 65.8%
    - Combines computation + context
    """)

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
    value="The decrease in Investment Banking segment revenue was primarily driven by lower trading volumes.",
    height=80,
)

if st.button("Run Hybrid Model", type="primary", use_container_width=True):
    st.divider()
    st.subheader("Response")
    st.write_stream(stream_hybrid(question, table, context))
