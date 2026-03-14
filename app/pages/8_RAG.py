"""
RAG Pipeline - Standalone query page
Runs real retrieval (sentence-transformers + ChromaDB) over financial documents,
then generates with Mistral-7B via Ollama.
URL: http://localhost:8501/RAG
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils import stream_rag, get_demo_status, LLM_MODEL, rag_num_chunks

st.set_page_config(page_title="RAG Pipeline", page_icon="FT", layout="wide")

st.title("RAG Pipeline Query")
st.markdown(
    f"Retrieval-Augmented Generation: embed the query with "
    "**sentence-transformers** (all-MiniLM-L6-v2), retrieve from **ChromaDB**, "
    f"then generate with **{LLM_MODEL}** (7B params) via Ollama."
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
    **Embedder:** all-MiniLM-L6-v2 (22M params)
    **Vector DB:** ChromaDB (in-memory, cosine)
    **Documents:** 12 financial reports

    ---

    **How it works:**
    1. Embed query into a 384-dim vector
    2. Search ChromaDB for top-3 similar chunks
    3. Augment prompt with retrieved documents
    4. {LLM_MODEL} generates answer from context

    **Key limitation:**
    The model's weights are NOT updated.
    It can only use what it retrieves -- it
    cannot perform reasoning it hasn't seen.
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

if st.button("Run RAG Pipeline", type="primary", use_container_width=True):
    st.divider()
    st.subheader("Response")
    st.write_stream(stream_rag(question, table, context))
