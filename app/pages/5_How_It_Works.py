"""
How It Works - Architecture Explainer
Technical overview of the three approaches
"""
import streamlit as st

st.set_page_config(page_title="How It Works", page_icon="FT", layout="wide")

st.title("How It Works")
st.markdown("Technical architecture of Fine-Tuned, RAG, and Hybrid approaches.")

# Three columns for architectures
tab1, tab2, tab3 = st.tabs(["Fine-Tuned", "RAG", "Hybrid"])

with tab1:
    st.subheader("Fine-Tuned Model Architecture")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    FINE-TUNED APPROACH                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Table + Text + Question                              │
    │         ↓                                                    │
    │  ┌─────────────────────────────────────────────┐            │
    │  │           FinQA-7B-Instruct                  │            │
    │  │  (Pre-trained on 8,281 financial Q&A pairs) │            │
    │  │                                              │            │
    │  │  • Learned numerical reasoning              │            │
    │  │  • Understands financial tables             │            │
    │  │  • Can perform calculations                 │            │
    │  └─────────────────────────────────────────────┘            │
    │         ↓                                                    │
    │  Output: Step-by-step reasoning + Answer                     │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **No retrieval step** - all knowledge is in model weights
    - **Fast inference** - single forward pass (~200ms)
    - **Learned reasoning** - can actually compute answers
    - **Consistent format** - trained for structured output

    **Training Data:**
    - FinQA dataset: 8,281 Q&A pairs from SEC filings
    - Each example includes table, text, question, and reasoning program
    """)

with tab2:
    st.subheader("RAG Architecture")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                      RAG APPROACH                            │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Question                                             │
    │         ↓                                                    │
    │  ┌─────────────────────┐                                    │
    │  │   Embedding Model   │  (all-MiniLM-L6-v2)                │
    │  └─────────────────────┘                                    │
    │         ↓                                                    │
    │  ┌─────────────────────┐                                    │
    │  │    Vector Store     │  (ChromaDB)                        │
    │  │  ┌───┐ ┌───┐ ┌───┐ │  100+ SEC filing chunks            │
    │  │  │ • │ │ • │ │ • │ │                                     │
    │  │  └───┘ └───┘ └───┘ │                                     │
    │  └─────────────────────┘                                    │
    │         ↓ (Top-K similar docs)                               │
    │  ┌─────────────────────┐                                    │
    │  │   Mistral-7B-Inst   │  General-purpose LLM               │
    │  │   + Retrieved Docs  │                                     │
    │  └─────────────────────┘                                    │
    │         ↓                                                    │
    │  Output: Generated answer (may struggle with math)           │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **Two-step process** - retrieve then generate
    - **Dynamic knowledge** - can access fresh documents
    - **Slower inference** - retrieval adds latency (~800ms)
    - **Cannot compute** - LLM not trained for numerical reasoning

    **Knowledge Base:**
    - SEC 10-K filings (sampled for demo)
    - Financial glossary and formulas
    - Embedding dimension: 384
    """)

with tab3:
    st.subheader("Hybrid Architecture")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                     HYBRID APPROACH                          │
    │              (Best of Both Worlds)                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Table + Text + Question                              │
    │         ↓                          ↓                         │
    │  ┌─────────────────────┐   ┌─────────────────────┐          │
    │  │   Embedding Model   │   │  Primary Context    │          │
    │  └─────────────────────┘   │  (Table + Text)     │          │
    │         ↓                  └─────────────────────┘          │
    │  ┌─────────────────────┐            │                       │
    │  │    Vector Store     │            │                       │
    │  │  (Retrieved Docs)   │            │                       │
    │  └─────────────────────┘            │                       │
    │         ↓                           ↓                        │
    │         └───────────┬───────────────┘                       │
    │                     ↓                                        │
    │  ┌─────────────────────────────────────────────┐            │
    │  │           FinQA-7B-Instruct                  │            │
    │  │   + Primary Data + Retrieved Context        │            │
    │  │                                              │            │
    │  │  • Domain expertise from fine-tuning        │            │
    │  │  • Fresh context from retrieval             │            │
    │  │  • Can perform calculations                 │            │
    │  └─────────────────────────────────────────────┘            │
    │         ↓                                                    │
    │  Output: Enriched reasoning + Accurate answer                │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **Best accuracy** - combines domain expertise with context
    - **Medium latency** - retrieval + specialized generation (~450ms)
    - **Can compute** - uses fine-tuned model for generation
    - **Context-aware** - can reference retrieved information

    **Use Case:**
    Ideal for production systems where accuracy is paramount and
    you need both computational ability and access to specific documents.
    """)

st.divider()

# When to use which
st.subheader("Decision Matrix: When to Use Each Approach")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Use Fine-Tuned When:
    - Task requires calculations
    - Low latency is critical
    - Output format must be consistent
    - Domain knowledge is stable
    - No need for source citations
    """)

with col2:
    st.markdown("""
    ### Use RAG When:
    - Data changes frequently
    - Need source citations for audit
    - No domain-specific training data
    - Broad knowledge required
    - Explainability via retrieval
    """)

with col3:
    st.markdown("""
    ### Use Hybrid When:
    - Maximum accuracy needed
    - Complex analysis required
    - Combining computation + context
    - Production systems
    - Can afford higher latency
    """)

st.divider()

st.subheader("Technical Stack")

tech_col1, tech_col2 = st.columns(2)

with tech_col1:
    st.markdown("""
    **Models:**
    - `truocpham/FinQA-7B-Instruct` - Fine-tuned for numerical reasoning
    - `ProsusAI/finbert` - Financial sentiment classification
    - `mistralai/Mistral-7B-Instruct-v0.2` - Base model for RAG

    **Frameworks:**
    - PyTorch + Transformers
    - LangChain + ChromaDB
    - Streamlit
    """)

with tech_col2:
    st.markdown("""
    **Datasets:**
    - FinQA: 8,281 Q&A pairs (IBM Research)
    - Financial PhraseBank: 5,000 sentences
    - SEC 10-K Filings (PleIAs/SEC)

    **Hardware:**
    - GPU recommended (NVIDIA 8GB+ VRAM)
    - 4-bit quantization for memory efficiency
    - Can run on CPU (slower)
    """)

st.divider()
st.caption("Architecture documentation for Fine-Tuning vs RAG demo")
