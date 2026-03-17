"""
How It Works - Architecture Explainer
Technical overview of the three experiments and their approaches
"""
import streamlit as st

st.set_page_config(page_title="How It Works", page_icon="FT", layout="wide")

st.title("How It Works")
st.markdown("Technical architecture behind each of the four benchmark experiments.")

# ==========================================================================
# Experiment 1: BERT 110M Sentiment
# ==========================================================================
st.header("Experiment 1: BERT 110M -- Sentiment Classification")
st.caption("Architecture: BERT-base-uncased (110M parameters)")

tab1a, tab1b, tab1c, tab1d = st.tabs(
    ["Base BERT", "FinBERT (Fine-Tuned)", "BERT + RAG", "FinBERT + RAG (Hybrid)"]
)

with tab1a:
    st.subheader("Base BERT -- No Fine-Tuning, No RAG")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │                  BASE BERT APPROACH                          │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Financial sentence                                   │
    │         ↓                                                    │
    │  ┌──────────────────────────────────────────────┐            │
    │  │        bert-base-uncased (110M params)       │            │
    │  │                                              │            │
    │  │  • General English language model            │            │
    │  │  • NOT trained on financial text             │            │
    │  │  • Assigns logits to 3 classes               │            │
    │  └──────────────────────────────────────────────┘            │
    │         ↓                                                    │
    │  Output: positive / negative / neutral + confidence          │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **Generic model** -- no financial domain knowledge
    - **Fast inference** -- single forward pass (~5ms)
    - **Struggles with jargon** -- e.g. "headwinds", "margin compression"
    - **Baseline** -- this is what we compare against
    """)

with tab1b:
    st.subheader("FinBERT -- Fine-Tuned on Financial Text")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │                 FINBERT (FINE-TUNED) APPROACH                │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Financial sentence                                   │
    │         ↓                                                    │
    │  ┌──────────────────────────────────────────────┐            │
    │  │         ProsusAI/finbert (110M params)       │            │
    │  │  (Fine-tuned on Financial PhraseBank)        │            │
    │  │                                              │            │
    │  │  • Learned financial sentiment patterns      │            │
    │  │  • Understands domain jargon                 │            │
    │  │  • Knows "declining cost ratio" = positive   │            │
    │  └──────────────────────────────────────────────┘            │
    │         ↓                                                    │
    │  Output: positive / negative / neutral + confidence          │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **Domain expertise baked in** -- all knowledge is in model weights
    - **Fast inference** -- single forward pass (~5ms)
    - **Handles domain inversions** -- "declining cost-to-income" = positive
    - **Same architecture** as Base BERT, different weights

    **Training Data:**
    - Financial PhraseBank: ~5,000 financial sentences
    - Labeled by financial analysts for sentiment
    """)

with tab1c:
    st.subheader("BERT + RAG -- Retrieval-Augmented Voting")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │                  BERT + RAG APPROACH                         │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Financial sentence                                   │
    │         ↓                                                    │
    │  ┌─────────────────────┐                                     │
    │  │   Embedding Model   │  (all-MiniLM-L6-v2)                 │
    │  └─────────────────────┘                                     │
    │         ↓                                                    │
    │  ┌─────────────────────┐                                     │
    │  │    Vector Store     │  (ChromaDB)                         │
    │  │  Similar financial  │  sentences with known labels        │
    │  └─────────────────────┘                                     │
    │         ↓ (Top-K similar examples)                           │
    │  ┌─────────────────────────────────────────┐                 │
    │  │       Majority Vote                     │                 │
    │  │  Labels of retrieved similar sentences  │                 │
    │  └─────────────────────────────────────────┘                 │
    │         ↓                                                    │
    │  Output: positive / negative / neutral + confidence          │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **No fine-tuning needed** -- uses base BERT embeddings + retrieval
    - **Dynamic knowledge** -- add new labeled examples anytime
    - **Majority voting** -- sentiment of nearest neighbors decides
    - **Slower** -- retrieval adds latency (~15ms)
    """)

with tab1d:
    st.subheader("FinBERT + RAG -- Hybrid Approach")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │               FINBERT + RAG (HYBRID) APPROACH                │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Financial sentence                                   │
    │         ↓                          ↓                         │
    │  ┌──────────────────┐    ┌─────────────────────┐             │
    │  │     FinBERT      │    │   Embedding Model   │             │
    │  │  (fine-tuned     │    │  + Vector Store     │             │
    │  │   prediction)    │    │  (retrieved labels) │             │
    │  └──────────────────┘    └─────────────────────┘             │
    │         ↓                          ↓                         │
    │         └──────────┬───────────────┘                         │
    │                    ↓                                         │
    │  ┌──────────────────────────────────────────────┐            │
    │  │          Weighted Combination                │            │
    │  │  FinBERT prediction + RAG neighbor vote      │            │
    │  └──────────────────────────────────────────────┘            │
    │         ↓                                                    │
    │  Output: positive / negative / neutral + confidence          │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **Best of both** -- domain expertise + retrieval evidence
    - **Weighted decision** -- FinBERT confidence + neighbor agreement
    - **Medium latency** -- FinBERT pass + retrieval (~20ms)
    - **Robust** -- fallback when either approach is uncertain
    """)

st.divider()

# ==========================================================================
# Experiment 2: Llama2 7B Numerical Reasoning
# ==========================================================================
st.header("Experiment 2: Llama2 7B -- Numerical Reasoning")
st.caption("Architecture: Llama2-7B (7B parameters) -- 5 multi-step calculation tasks")

tab2a, tab2b, tab2c, tab2d = st.tabs(
    ["Base Llama2-7B", "FinQA-7B (Fine-Tuned)", "Llama2-7B + RAG", "FinQA-7B + RAG (Hybrid)"]
)

with tab2a:
    st.subheader("Base Llama2-7B -- General-Purpose LLM")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │                  BASE LLAMA2-7B APPROACH                     │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Table + Context + Question                           │
    │         ↓                                                    │
    │  ┌──────────────────────────────────────────────┐            │
    │  │           Llama2-7B (base model)             │            │
    │  │                                              │            │
    │  │  • General language model                    │            │
    │  │  • NOT trained for financial math            │            │
    │  │  • Generic prompt                            │            │
    │  └──────────────────────────────────────────────┘            │
    │         ↓                                                    │
    │  Output: Attempted reasoning + Answer (often wrong)          │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **No domain training** -- relies on general pre-training
    - **Struggles with calculations** -- often hallucinates numbers
    - **Baseline** -- demonstrates why specialization matters
    """)

with tab2b:
    st.subheader("FinQA-7B -- Fine-Tuned for Financial Q&A")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │                    FINQA-7B APPROACH                         │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Table + Context + Question                           │
    │         ↓                                                    │
    │  ┌──────────────────────────────────────────────┐            │
    │  │           FinQA-7B-Instruct                  │            │
    │  │  (Llama2-7B + LoRA adapter fine-tuned on     │            │
    │  │   8,281 financial Q&A pairs from FinQA)      │            │
    │  │                                              │            │
    │  │  • Learned numerical reasoning               │            │
    │  │  • Understands financial tables              │            │
    │  │  • Can perform multi-step calculations       │            │
    │  └──────────────────────────────────────────────┘            │
    │         ↓                                                    │
    │  Output: Step-by-step reasoning + Answer                     │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **Domain expertise in weights** -- learned from 8,281 Q&A pairs
    - **LoRA adapter** -- only ~128MB on top of base Llama2-7B
    - **Can compute** -- trained to perform financial calculations
    - **Consistent output** -- trained for structured reasoning

    **Training Data:**
    - FinQA dataset: 8,281 Q&A pairs from SEC filings (IBM Research)
    - Each example: table + text + question + reasoning program
    """)

with tab2c:
    st.subheader("Llama2-7B + RAG -- Retrieval-Augmented Generation")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                   LLAMA2 + RAG APPROACH                     │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Input: Table + Context + Question                          │
    │         ↓                                                   │
    │  ┌─────────────────────┐                                    │
    │  │   Embedding Model   │  (all-MiniLM-L6-v2)                │
    │  └─────────────────────┘                                    │
    │         ↓                                                   │
    │  ┌─────────────────────┐                                    │
    │  │    Vector Store     │  (ChromaDB)                        │
    │  │  SEC filing chunks  │  + financial formulas              │
    │  └─────────────────────┘                                    │
    │         ↓ (Top-K similar docs)                              │
    │  ┌─────────────────────────────────────────────┐            │
    │  │   Llama2-7B (base) + Retrieved Documents    │            │
    │  │   General LLM with added context            │            │
    │  └─────────────────────────────────────────────┘            │
    │         ↓                                                   │
    │  Output: Generated answer (may struggle with math)          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **Two-step process** -- retrieve relevant docs, then generate
    - **Dynamic knowledge** -- can access fresh documents
    - **Slower inference** -- retrieval adds latency
    - **Cannot compute well** -- base LLM not trained for math

    **Knowledge Base:**
    - SEC 10-K filings (sampled for demo)
    - Financial glossary and formulas
    - 24 chunks indexed, embedding dimension: 384
    """)

with tab2d:
    st.subheader("FinQA-7B + RAG -- Hybrid Approach")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                     HYBRID APPROACH                         │
    │              (Best of Both Worlds)                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Input: Table + Context + Question                          │
    │         ↓                          ↓                        │
    │  ┌─────────────────────┐   ┌─────────────────────┐          │
    │  │   Embedding Model   │   │  Primary Context    │          │
    │  └─────────────────────┘   │  (Table + Text)     │          │
    │         ↓                  └─────────────────────┘          │
    │  ┌─────────────────────┐            │                       │
    │  │    Vector Store     │            │                       │
    │  │  (Retrieved Docs)   │            │                       │
    │  └─────────────────────┘            │                       │
    │         ↓                           ↓                       │
    │         └───────────┬───────────────┘                       │
    │                     ↓                                       │
    │  ┌──────────────────────────────────────────────┐           │
    │  │           FinQA-7B-Instruct                  │           │
    │  │   + Primary Data + Retrieved Context         │           │
    │  │                                              │           │
    │  │  • Domain expertise from fine-tuning         │           │
    │  │  • Fresh context from retrieval              │           │
    │  │  • Can perform calculations                  │           │
    │  └──────────────────────────────────────────────┘           │
    │         ↓                                                   │
    │  Output: Enriched reasoning + Accurate answer               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **Best accuracy** -- combines domain expertise with context
    - **Can compute** -- uses fine-tuned model for generation
    - **Context-aware** -- can reference retrieved information
    - **Higher latency** -- retrieval + specialized generation
    """)

st.divider()

# ==========================================================================
# Experiment 3: Llama2 7B Financial Ratios
# ==========================================================================
st.header("Experiment 3: Llama2 7B -- Financial Ratios")
st.caption("Architecture: Same as Experiment 2 -- 8 complex multi-step ratio calculations")

st.markdown("""
This experiment uses the **same four approaches** as Experiment 2 (Base Llama2-7B, FinQA-7B,
Llama2-7B + RAG, FinQA-7B + RAG) but on harder tasks:

| Aspect | Experiment 2 (Numerical) | Experiment 3 (Financial Ratios) |
|---|---|---|
| **Test cases** | 5 cases | 8 cases |
| **Complexity** | Single/multi-step arithmetic | Multi-step ratio decomposition |
| **Examples** | % change, D/E ratio | DuPont ROE, CAGR, sustainable growth rate |
| **Categories** | single_step, multi_step | profitability, efficiency, liquidity, leverage, shareholder |
| **Why separate** | Tests basic math ability | Tests whether the model can chain multiple formulas |

The architectural diagrams are identical to Experiment 2 -- the difference is in what we ask each model to compute.
""")

st.divider()

# ==========================================================================
# Experiment 4: DistilBERT 66M Spam Detection
# ==========================================================================
st.header("Experiment 4: DistilBERT 66M -- Spam Detection")
st.caption("Architecture: DistilBERT-base-uncased (66M parameters) -- 20 spam/ham email classification tasks")

tab4a, tab4b, tab4c, tab4d = st.tabs(
    ["Base DistilBERT", "Fine-Tuned DistilBERT (Spam)", "DistilBERT + RAG", "Fine-Tuned + RAG (Hybrid)"]
)

with tab4a:
    st.subheader("Base DistilBERT -- No Fine-Tuning, No RAG")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │              BASE DISTILBERT APPROACH                        │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Email text                                           │
    │         ↓                                                    │
    │  ┌──────────────────────────────────────────────┐            │
    │  │     distilbert-base-uncased (66M params)     │            │
    │  │                                              │            │
    │  │  • General English language model            │            │
    │  │  • NOT trained on spam/phishing data         │            │
    │  │  • Zero-shot cosine similarity to prototypes │            │
    │  └──────────────────────────────────────────────┘            │
    │         ↓                                                    │
    │  Output: spam / ham + confidence                             │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **Generic model** -- no spam/phishing domain knowledge
    - **Fast inference** -- single forward pass (~5ms)
    - **Struggles with subtle phishing** -- may miss urgency cues
    - **Baseline** -- this is what we compare against
    """)

with tab4b:
    st.subheader("Fine-Tuned DistilBERT -- Trained on Spam/Phishing Data")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │           FINE-TUNED DISTILBERT APPROACH                     │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Email text                                           │
    │         ↓                                                    │
    │  ┌──────────────────────────────────────────────┐            │
    │  │  Fine-tuned DistilBERT (66M params)          │            │
    │  │  (Trained on phishing/spam dataset)           │            │
    │  │                                              │            │
    │  │  • Learned spam/phishing patterns            │            │
    │  │  • Recognizes urgency, suspicious URLs       │            │
    │  │  • Knows "verify your account" = phishing    │            │
    │  └──────────────────────────────────────────────┘            │
    │         ↓                                                    │
    │  Output: spam / ham + confidence                             │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **Spam expertise baked in** -- all knowledge is in model weights
    - **Fast inference** -- single forward pass (~5ms)
    - **Handles phishing patterns** -- urgency, fake verification, prize scams
    - **Same architecture** as Base DistilBERT, different weights

    **Training Data:**
    - 10,000 labeled emails (phishing + legitimate)
    - Binary classification: spam (phishing) vs ham (legitimate)

    **Source:**
    - [enterprise-mailbox-assistant](https://github.com/intelliswarm-ai/enterprise-mailbox-assistant/tree/main/model-fine-tuned-llm)
    """)

with tab4c:
    st.subheader("DistilBERT + RAG -- Retrieval-Augmented Voting")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │               DISTILBERT + RAG APPROACH                      │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Email text                                           │
    │         ↓                                                    │
    │  ┌─────────────────────┐                                     │
    │  │   Embedding Model   │  (distilbert-base-uncased)          │
    │  └─────────────────────┘                                     │
    │         ↓                                                    │
    │  ┌─────────────────────┐                                     │
    │  │  Knowledge Base     │  15 labeled spam/ham examples       │
    │  │  (Cosine similarity │  for retrieval)                     │
    │  └─────────────────────┘                                     │
    │         ↓ (Top-5 similar examples)                           │
    │  ┌─────────────────────────────────────────┐                 │
    │  │     Similarity-Weighted Vote            │                 │
    │  │  Labels of retrieved similar emails     │                 │
    │  └─────────────────────────────────────────┘                 │
    │         ↓                                                    │
    │  Output: spam / ham + confidence                             │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **No fine-tuning needed** -- uses base DistilBERT embeddings + retrieval
    - **Dynamic knowledge** -- add new labeled examples anytime
    - **Similarity voting** -- nearest neighbors decide the label
    - **Slower** -- retrieval adds latency (~15ms)
    """)

with tab4d:
    st.subheader("Fine-Tuned + RAG -- Hybrid Approach")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │           FINE-TUNED + RAG (HYBRID) APPROACH                 │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Input: Email text                                           │
    │         ↓                          ↓                         │
    │  ┌──────────────────┐    ┌─────────────────────┐             │
    │  │  Fine-Tuned      │    │   Embedding Model   │             │
    │  │  DistilBERT      │    │  + Knowledge Base   │             │
    │  │  (spam-trained   │    │  (retrieved labels) │             │
    │  │   prediction)    │    │                     │             │
    │  └──────────────────┘    └─────────────────────┘             │
    │         ↓                          ↓                         │
    │         └──────────┬───────────────┘                         │
    │                    ↓                                         │
    │  ┌──────────────────────────────────────────────┐            │
    │  │          Weighted Combination                │            │
    │  │  60% Fine-Tuned + 40% RAG neighbor vote     │            │
    │  └──────────────────────────────────────────────┘            │
    │         ↓                                                    │
    │  Output: spam / ham + confidence                             │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Key Characteristics:**
    - **Best of both** -- spam expertise + retrieval evidence
    - **Weighted decision** -- fine-tuned confidence + neighbor agreement
    - **Medium latency** -- fine-tuned pass + retrieval (~20ms)
    - **Robust** -- fallback when either approach is uncertain
    """)

st.divider()

# ==========================================================================
# Decision Matrix
# ==========================================================================
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
    - `ProsusAI/finbert` -- Fine-tuned BERT for sentiment (110M)
    - `bert-base-uncased` -- Base BERT baseline (110M)
    - `truocpham/FinQA-7B-Instruct` -- Fine-tuned Llama2 for financial Q&A (7B)
    - `llama2` -- Base Llama2 via Ollama (7B)
    - `distilbert-base-uncased` -- Base DistilBERT baseline (66M)
    - Fine-tuned DistilBERT -- Spam/phishing detector (66M)

    **Frameworks:**
    - PyTorch + Transformers (BERT/DistilBERT models)
    - Ollama (Llama2 inference)
    - ChromaDB + sentence-transformers (RAG)
    - Streamlit (UI)
    """)

with tech_col2:
    st.markdown("""
    **Datasets:**
    - FinQA: 8,281 Q&A pairs (IBM Research)
    - Financial PhraseBank: ~5,000 sentences
    - SEC 10-K Filings (sample documents for RAG)
    - Phishing/Spam: 10,000 labeled emails (spam detection)

    **Infrastructure:**
    - Docker Compose (demo + Ollama containers)
    - GPU recommended for Llama2 (can run on CPU)
    - FinBERT runs on CPU (~5ms/query)
    - Embedding: all-MiniLM-L6-v2 (384 dimensions)
    """)

st.divider()
st.caption("Architecture documentation for Fine-Tuning vs RAG demo -- four experiments across three architectures")
