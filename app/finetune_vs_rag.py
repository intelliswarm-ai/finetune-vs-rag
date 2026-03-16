"""
Fine-Tuning vs RAG: Financial Services Demo
Main Streamlit Application - Landing Page
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Fine-Tuning vs RAG Demo",
    page_icon="FT",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .approach-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem;
        height: 100%;
    }
    .finetuned-card { border-color: #28a745; background-color: #f8fff8; }
    .rag-card { border-color: #007bff; background-color: #f8f9ff; }
    .hybrid-card { border-color: #9b59b6; background-color: #faf8ff; }
    .flow-step {
        background-color: #f0f2f6;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 4px solid #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Fine-Tuning vs RAG: Live Demo</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive presentation and live comparison for financial AI</p>', unsafe_allow_html=True)
st.markdown("[:material/code: GitHub Repository](https://github.com/intelliswarm-ai/finetune-vs-rag)")

# Check demo status
try:
    from demo_utils import get_demo_status
    status = get_demo_status()
except Exception:
    status = {}

# Status banner
if status.get("openai") == "live" or status.get("finbert") == "live":
    live_parts = [k for k, v in status.items() if v == "live"]
    st.success(f"[LIVE] **Live mode active** - {', '.join(live_parts)} connected")
else:
    st.info("[SIM] **Demo mode** - Using simulated responses. Add API keys to .env for live inference.")

st.divider()

# Quick Navigation
st.subheader("Quick Navigation")
if st.button("Start Presentation", type="primary", use_container_width=True):
    st.switch_page("pages/0_Presentation.py")
if st.button("Sentiment Demo", use_container_width=True):
    st.switch_page("pages/3_Sentiment_Analysis.py")
if st.button("Numerical Reasoning Demo", use_container_width=True):
    st.switch_page("pages/1_Numerical_Reasoning.py")
if st.button("Benchmark Results", use_container_width=True):
    st.switch_page("pages/4_Benchmark_Results.py")

st.divider()

# Three approaches overview
st.subheader("Three Approaches Compared")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="approach-card finetuned-card">
        <h3>Fine-Tuned</h3>
        <p><strong>Model:</strong> FinQA-7B / FinBERT</p>
        <p><strong>Best for:</strong></p>
        <ul>
            <li>Numerical calculations</li>
            <li>Domain-specific reasoning</li>
            <li>Consistent output format</li>
            <li>Low latency (~200ms)</li>
        </ul>
        <p><strong>Key:</strong> Teaches NEW SKILLS</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="approach-card rag-card">
        <h3>RAG</h3>
        <p><strong>Model:</strong> Any LLM + Vector DB</p>
        <p><strong>Best for:</strong></p>
        <ul>
            <li>Fresh/dynamic data</li>
            <li>Source citations</li>
            <li>No training required</li>
            <li>Quick deployment</li>
        </ul>
        <p><strong>Key:</strong> Adds INFORMATION</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="approach-card hybrid-card">
        <h3>Hybrid</h3>
        <p><strong>Model:</strong> Fine-Tuned + RAG</p>
        <p><strong>Best for:</strong></p>
        <ul>
            <li>Maximum accuracy</li>
            <li>Complex analysis</li>
            <li>Production systems</li>
            <li>Best of both worlds</li>
        </ul>
        <p><strong>Key:</strong> SKILLS + INFORMATION</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Key metrics
st.subheader("Performance Summary")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        label="Numerical Accuracy",
        value="FT: 61.2%",
        delta="+46% vs RAG (15.3%)",
    )
with m2:
    st.metric(
        label="Sentiment Accuracy",
        value="FinBERT: 94%",
        delta="+16% vs RAG",
    )
with m3:
    st.metric(
        label="Inference Speed",
        value="FT: ~200ms",
        delta="4x faster than RAG",
    )
with m4:
    st.metric(
        label="Output Consistency",
        value="FT: 98%",
        delta="+33% vs RAG (65%)",
    )

# Footer
st.divider()
st.markdown("""
> *"Fine-tuning teaches the model the **language of your domain** - not just definitions,
> but how to **reason** within it."*

---
**Presentation Demo** | Models: FinQA-7B, FinBERT, Mistral-7B | Data: FinQA, Financial PhraseBank, SEC Filings
""")

# Sidebar
with st.sidebar:
    st.header("About This Demo")
    st.markdown("""
    **Topic:** LLM Fine-Tuning vs RAG

    **Core Message:**
    Fine-tuning teaches **skills**.
    RAG provides **information**.
    The hybrid approach gives **both**.

    ---

    **Pages:**
    """)
    st.page_link("Finetune_vs_RAG.py", label="Home")
    st.page_link("pages/0_Presentation.py", label="Presentation Slides")
    st.page_link("pages/1_Numerical_Reasoning.py", label="Numerical Reasoning")
    st.page_link("pages/2_Financial_Ratios.py", label="Financial Ratios")
    st.page_link("pages/3_Sentiment_Analysis.py", label="Sentiment Analysis")
    st.page_link("pages/4_Benchmark_Results.py", label="Benchmark Results")
    st.page_link("pages/5_How_It_Works.py", label="How It Works")

    st.divider()
    st.markdown("**Live Query (own URL/tab):**")
    st.page_link("pages/6_Live_Query.py", label="All Models Side-by-Side")
    st.page_link("pages/7_Fine_Tuned.py", label="Fine-Tuned Only")
    st.page_link("pages/8_RAG.py", label="RAG Only")
    st.page_link("pages/9_Hybrid.py", label="Hybrid Only")

    st.divider()

    st.markdown("### Demo Status")
    for component, state in status.items():
        if state == "live":
            st.success(f"[LIVE] {component}")
        else:
            st.warning(f"[SIM] {component} (simulated)")

    if not status:
        st.info("Run from project root for status detection")
