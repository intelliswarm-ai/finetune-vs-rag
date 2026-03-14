"""
Fine-Tuned Model - Standalone query page
Runs Mistral-7B with a domain-expert system prompt via Ollama.
In production this would be a truly fine-tuned model (e.g. FinQA-7B).
URL: http://localhost:8501/Fine_Tuned
"""
import streamlit as st
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_utils import stream_finetuned, get_demo_status, LLM_MODEL

st.set_page_config(page_title="Fine-Tuned Model", page_icon="FT", layout="wide")

st.title("Fine-Tuned Model Query")
st.markdown(
    f"Running **{LLM_MODEL}** (7B params) with a financial-expert system prompt. "
    "In production, this would be a model whose weights were updated by "
    "training on domain data (e.g. FinQA-7B trained on 8,281 financial Q&A pairs)."
)

with st.sidebar:
    st.header("Model Card")
    status = get_demo_status()
    if status.get("ollama") == "live":
        st.success(f"[LIVE] {LLM_MODEL}")
    else:
        st.error("[OFFLINE] Ollama not reachable")

    st.markdown(f"""
    **Model:** {LLM_MODEL}
    **Parameters:** ~7B
    **Approach:** Domain-expert system prompt
    (approximates fine-tuned behavior)

    ---

    **What true fine-tuning does:**
    - Updates model weights on domain data
    - Learns numerical reasoning patterns
    - Learns financial vocabulary
    - Single forward pass, no retrieval

    **Benchmark (FinQA-7B, actually fine-tuned):**
    - Execution accuracy: 61.2%
    - Latency: ~200ms
    - Output consistency: 98%
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

if st.button("Run Fine-Tuned Model", type="primary", use_container_width=True):
    st.divider()
    st.subheader("Response")
    st.write_stream(stream_finetuned(question, table, context))
