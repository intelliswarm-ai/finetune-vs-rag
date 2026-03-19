"""
Interactive Presentation Slides
LLM Fine-Tuning: Maximizing AI for Specialized Tasks
Navigate with Previous/Next buttons or keyboard shortcuts.
"""
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Presentation - Fine-Tuning vs RAG",
    page_icon="FT",
    layout="wide",
)


def render_mermaid(diagram_code, height=500):
    """Render a Mermaid diagram in a fixed-height iframe."""
    html = f"""
    <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; }}
        .mermaid {{ display: flex; justify-content: center; align-items: start; }}
        .mermaid svg {{ max-width: 100%; height: auto; }}
    </style>
    <div class="mermaid">
    {diagram_code}
    </div>
    <script>
        mermaid.initialize({{startOnLoad: true, theme: 'neutral', securityLevel: 'loose'}});
    </script>
    """
    components.html(html, height=height, scrolling=False)


# ---------------------------------------------------------------------------
# Custom CSS for presentation look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Slide container */
    .slide-container {
        min-height: 500px;
        padding: 1rem 0;
    }
    .slide-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 0.5rem;
    }
    .slide-subtitle {
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .big-title {
        font-size: 3rem;
        font-weight: bold;
        color: #0066cc;
        text-align: center;
        margin-top: 2rem;
    }
    .big-subtitle {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .highlight-box {
        background: linear-gradient(135deg, #1a56db 0%, #1e3a5f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #ffffff;
        margin: 1rem 0;
    }
    .highlight-box strong, .highlight-box b { color: #ffffff; }
    .green-box {
        background-color: #ecfdf5;
        border-left: 5px solid #16a34a;
        padding: 1rem 1.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #14532d;
    }
    .green-box strong, .green-box b { color: #14532d; }
    .red-box {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 1rem 1.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #7f1d1d;
    }
    .red-box strong, .red-box b { color: #7f1d1d; }
    .blue-box {
        background-color: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 1rem 1.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #1e3a5f;
    }
    .blue-box strong, .blue-box b { color: #1e3a5f; }
    .orange-box {
        background-color: #fffbeb;
        border-left: 5px solid #d97706;
        padding: 1rem 1.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #78350f;
    }
    .orange-box strong, .orange-box b { color: #78350f; }
    .tool-card {
        border: 2px solid #d1d5db;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9fafb;
        color: #1f2937;
    }
    .comparison-table th {
        background-color: #1a56db;
        color: #ffffff;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a56db;
    }
    .section-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Slide definitions
# ---------------------------------------------------------------------------


def slide_title():
    """Slide 0: Title slide"""
    st.markdown("")
    st.markdown("")
    st.markdown('<p class="big-title">LLM Fine-Tuning</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-subtitle">Maximizing AI Performance for Specialized Tasks</p>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Topics:**")
        st.markdown("Fine-Tuning vs RAG vs Hybrid")
    with col2:
        st.markdown("**Audience:**")
        st.markdown("No deep technical experience required")
    with col3:
        st.markdown("**Format:**")
        st.markdown("Presentation + Live Demo")

    st.markdown("---")
    st.markdown("""
    > *"Fine-tuning teaches a model new **skills**. RAG gives it new **information**.
    > Knowing the difference is the key to building effective AI systems."*
    """)


def slide_agenda():
    """Slide 1: Agenda"""
    st.markdown('<p class="slide-title">Agenda</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Part 1: Understanding the Landscape
        1. What are Large Language Models (LLMs)?
        2. The specialization challenge
        3. Three approaches to domain AI

        ### Part 2: Deep Dive
        4. RAG - Benefits & limitations
        5. Fine-Tuning - Benefits & methods
        6. Head-to-head comparison
        7. Decision framework
        """)
    with col2:
        st.markdown("""
        ### Part 3: Tools & Ecosystem
        8. Fine-tuning tools & platforms
        9. RAG tools & infrastructure
        10. The hybrid approach

        ### Part 4: Live Demo
        11. Sentiment analysis comparison
        12. Financial reasoning comparison
        13. Spam detection comparison
        14. Benchmark results
        15. Key takeaways & Q&A
        """)


def slide_what_are_llms():
    """Slide 2: What are LLMs"""
    st.markdown('<p class="slide-title">What Are Large Language Models?</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### How LLMs Work
        - Trained on **massive text datasets** (books, web, code)
        - Learn language **patterns**, **facts**, and **reasoning**
        - Billions of parameters encode learned knowledge
        - Generate text by predicting the next token

        ### Key Characteristics
        - **General-purpose** by design - know a lot about everything
        - Can follow instructions, answer questions, write code
        - Powerful but **not specialized** for any single domain
        """)
    with col2:
        st.markdown("""
        ### Popular LLMs
        | Model | Creator | Parameters |
        |-------|---------|-----------|
        | GPT-4 | OpenAI | ~1.7T |
        | Claude | Anthropic | Undisclosed |
        | Llama 3 | Meta | 8B-405B |
        | Mistral | Mistral AI | 7B-141B |
        | Gemini | Google | Undisclosed |
        """)

    st.info("**Think of an LLM as a brilliant generalist** - it can discuss any topic but isn't an expert in any single domain.")


def slide_specialization_challenge():
    """Slide 3: The Challenge"""
    st.markdown('<p class="slide-title">The Specialization Challenge</p>', unsafe_allow_html=True)

    st.markdown("""
    ### The Problem: Generic Models Fall Short on Domain Tasks
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="red-box">
        <strong>What generic LLMs struggle with:</strong>
        <ul>
            <li>Precise financial calculations</li>
            <li>Industry-specific terminology</li>
            <li>Domain reasoning patterns</li>
            <li>Consistent output formats</li>
            <li>Regulatory/compliance accuracy</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="green-box">
        <strong>What domain experts need:</strong>
        <ul>
            <li>Accurate, verifiable answers</li>
            <li>Correct use of specialized vocabulary</li>
            <li>Step-by-step reasoning</li>
            <li>Consistent, auditable output</li>
            <li>Speed and reliability</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.warning("""
    **Example:** Ask a generic LLM to calculate a bank's efficiency ratio from a financial table.
    It may know the *definition* but often gets the *calculation* wrong or provides vague answers.
    """)


def slide_three_approaches():
    """Slide 4: Three Approaches"""
    st.markdown('<p class="slide-title">Three Approaches to Specialization</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 1. Prompt Engineering
        *Tell the model what to do*

        ---

        - Write better instructions
        - Add examples (few-shot)
        - System prompts

        **Effort:** Low
        **Impact:** Limited
        **Best for:** Quick experiments
        """)

    with col2:
        st.markdown("""
        ### 2. RAG
        *Give the model information*

        ---

        - Retrieve relevant documents
        - Augment the prompt
        - Generate with context

        **Effort:** Medium
        **Impact:** Medium
        **Best for:** Knowledge lookup
        """)

    with col3:
        st.markdown("""
        ### 3. Fine-Tuning
        *Teach the model new skills*

        ---

        - Train on domain data
        - Model weights are updated
        - Learns new capabilities

        **Effort:** Higher
        **Impact:** Highest
        **Best for:** Specialized tasks
        """)

    st.markdown("---")
    st.markdown("**Spectrum of complexity:** Prompt Engineering < RAG < Fine-Tuning < Training from Scratch")


def slide_rag_explained():
    """Slide 5: RAG Explained"""
    st.markdown('<p class="slide-title">RAG: Retrieval-Augmented Generation</p>', unsafe_allow_html=True)

    st.markdown("### How RAG Works")

    render_mermaid("""
    graph TD
        A["User Question"] --> B["<b>1. EMBED</b><br/>Convert question to vector"]
        B --> C["<b>2. RETRIEVE</b><br/>Search vector database<br/>for similar documents"]
        C --> D["<b>3. AUGMENT</b><br/>Add retrieved documents<br/>to the prompt"]
        D --> E["<b>4. GENERATE</b><br/>LLM generates answer using<br/>question + retrieved context"]
        E --> F["<b>Answer</b><br/>with source references"]

        style A fill:#f5f5f5,stroke:#424242,stroke-width:2px,color:#212121
        style B fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,color:#0d47a1
        style C fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,color:#0d47a1
        style D fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,color:#0d47a1
        style E fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
        style F fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#5d4037
    """, height=550)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Key Components
        - **Embedding Model**: Converts text to vectors (e.g., sentence-transformers)
        - **Vector Database**: Stores and searches document embeddings (e.g., ChromaDB, Pinecone)
        - **LLM**: Generates the final answer using retrieved context
        """)
    with col2:
        st.markdown("""
        ### Key Idea
        > "Instead of training the model on your data, you **show** it relevant
        > documents at inference time."

        The model's weights are **unchanged** - you're just adding context to the prompt.
        """)


def slide_rag_benefits():
    """Slide 6: RAG Benefits"""
    st.markdown('<p class="slide-title">RAG: Benefits</p>', unsafe_allow_html=True)

    benefits = [
        ("No Training Required", "Start using immediately with any LLM. No GPU, no training data preparation, no wait time."),
        ("Dynamic Knowledge", "Update the knowledge base anytime. Add new documents, remove outdated ones. No retraining needed."),
        ("Source Citations", "Every answer can reference the exact documents used. Critical for audit and compliance."),
        ("Cost Effective to Start", "Lower upfront cost. Use existing LLM APIs + a vector database."),
        ("Explainable", "Users can see which documents were retrieved and why the model answered a certain way."),
    ]

    for i, (title, desc) in enumerate(benefits):
        st.markdown(f"""
        <div class="green-box">
        <strong>{i+1}. {title}</strong><br/>
        {desc}
        </div>
        """, unsafe_allow_html=True)


def slide_rag_limitations():
    """Slide 7: RAG Limitations (KEY SLIDE)"""
    st.markdown('<p class="slide-title">RAG: Limitations</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">Why RAG may not fully address all use cases</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="red-box">
        <strong>1. Cannot Teach New Skills</strong><br/>
        RAG adds <em>information</em> but doesn't teach the model new <em>reasoning patterns</em>.
        A model that can't do math won't learn math from retrieved documents.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="red-box">
        <strong>2. Struggles with Complex Calculations</strong><br/>
        Retrieving a formula doesn't mean the model can apply it correctly.
        Multi-step financial calculations frequently fail with RAG.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="red-box">
        <strong>3. Inconsistent Output</strong><br/>
        Without trained behavior patterns, output format varies between queries.
        Critical for production systems that need structured responses.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="red-box">
        <strong>4. Retrieval Quality Bottleneck</strong><br/>
        Answers are only as good as the retrieved documents.
        Poor retrieval = poor answers ("garbage in, garbage out").
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="red-box">
        <strong>5. Higher Latency</strong><br/>
        Embedding + retrieval + generation adds significant latency.
        Typically 3-5x slower than a fine-tuned model for the same task.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="red-box">
        <strong>6. Context Window Limitations</strong><br/>
        Retrieved documents compete for space in the context window.
        Too much context can actually confuse the model.
        </div>
        """, unsafe_allow_html=True)

    st.error("**Bottom line:** RAG is great for knowledge lookup but insufficient for tasks requiring specialized reasoning, computation, or consistent behavior.")


def slide_finetuning_explained():
    """Slide 8: Fine-Tuning Explained"""
    st.markdown('<p class="slide-title">Fine-Tuning: Teaching Models New Skills</p>', unsafe_allow_html=True)

    st.markdown("### How Fine-Tuning Works")

    render_mermaid("""
    graph TD
        A["Domain-Specific Dataset<br/><i>e.g. 8,000+ financial Q&A pairs</i>"] --> B["<b>1. PREPARE</b><br/>Format data as<br/>instruction / response pairs"]
        B --> C["<b>2. TRAIN</b><br/>Update model weights on your data<br/><i>Full fine-tuning or LoRA / QLoRA</i>"]
        C --> D["<b>3. EVALUATE</b><br/>Test on held-out data,<br/>measure accuracy"]
        D --> E["<b>4. DEPLOY</b><br/>Use the specialized model<br/>for inference"]
        E --> F["<b>Model with NEW capabilities</b><br/>Reasoning, calculations,<br/>domain expertise"]

        style A fill:#f5f5f5,stroke:#424242,stroke-width:2px,color:#212121
        style B fill:#fff3e0,stroke:#e65100,stroke-width:1px,color:#bf360c
        style C fill:#fff3e0,stroke:#e65100,stroke-width:1px,color:#bf360c
        style D fill:#fff3e0,stroke:#e65100,stroke-width:1px,color:#bf360c
        style E fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
        style F fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    """, height=550)

    st.markdown("""
    ### Key Difference from RAG
    > **RAG** = Same model + external information at query time
    > **Fine-Tuning** = **Different model** with learned domain expertise embedded in its weights
    """)

    st.success("The model doesn't just *look up* answers - it has *learned* how to reason about your domain.")


def slide_finetuning_methods():
    """Slide 9: Fine-Tuning Methods"""
    st.markdown('<p class="slide-title">Fine-Tuning Methods</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### Full Fine-Tuning
        - Updates **all** model parameters
        - Maximum learning capacity
        - Requires significant GPU memory
        - Best accuracy potential

        **When to use:**
        Large datasets, critical accuracy

        **Cost:** $$$
        """)

    with col2:
        st.markdown("""
        ### LoRA
        *Low-Rank Adaptation*
        - Adds small **adapter layers**
        - Freezes original weights
        - 10-100x less memory
        - Near full fine-tuning quality

        **When to use:**
        Most use cases (recommended)

        **Cost:** $$
        """)

    with col3:
        st.markdown("""
        ### QLoRA
        *Quantized LoRA*
        - LoRA + **4-bit quantization**
        - Runs on consumer GPUs
        - Minimal quality loss
        - Most accessible method

        **When to use:**
        Limited hardware, quick iteration

        **Cost:** $
        """)

    st.info("""
    **Practical tip:** Start with QLoRA for rapid experimentation, then scale to LoRA or full
    fine-tuning when you've validated the approach.
    """)


def slide_finetuning_benefits():
    """Slide 10: Fine-Tuning Benefits (KEY SLIDE)"""
    st.markdown('<p class="slide-title">Fine-Tuning: Key Benefits</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">Why fine-tuning is essential for specialized tasks</p>', unsafe_allow_html=True)

    benefits = [
        ("Improved Accuracy", "61.2% execution accuracy on FinQA vs 15.3% for RAG. Fine-tuned models learn domain-specific reasoning patterns."),
        ("Learned Computation", "Can perform multi-step calculations, not just retrieve formulas. The model actually 'understands' financial math."),
        ("Consistent Output", "98% output consistency vs 65% for RAG. Trained to produce structured, predictable responses every time."),
        ("Lower Latency", "~200ms vs ~800ms for RAG. No retrieval step needed - knowledge is in the model weights."),
        ("Customizable Behavior", "Control the model's tone, format, reasoning style, and domain vocabulary through training data."),
        ("Adaptability to Unique Data", "Train on your proprietary data. The model learns YOUR domain's patterns, terminology, and edge cases."),
    ]

    for i in range(0, len(benefits), 2):
        col1, col2 = st.columns(2)
        with col1:
            title, desc = benefits[i]
            st.markdown(f"""
            <div class="green-box">
            <strong>{title}</strong><br/>
            {desc}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if i + 1 < len(benefits):
                title, desc = benefits[i + 1]
                st.markdown(f"""
                <div class="green-box">
                <strong>{title}</strong><br/>
                {desc}
                </div>
                """, unsafe_allow_html=True)


def slide_models_used():
    """Models used in this demo and their training"""
    st.markdown('<p class="slide-title">Our Fine-Tuned Models: Under the Hood</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">Real models, real training data, published research</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="green-box">
        <strong>FinBERT -- Financial Sentiment Classification</strong><br/><br/>
        <strong>Base model:</strong> BERT-base-uncased (110M parameters)<br/>
        <strong>Fine-tuned by:</strong> Prosus AI (Dogu Araci, 2019)<br/>
        <strong>HuggingFace:</strong> <code>ProsusAI/finbert</code><br/><br/>
        <strong>What was changed:</strong> The final classification head was added
        and the entire model was further pre-trained on financial text, then
        fine-tuned for 3-class sentiment (positive, negative, neutral).<br/><br/>
        <strong>Paper:</strong> <em>"FinBERT: Financial Sentiment Analysis with
        Pre-Trained Language Models"</em> (2019)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="blue-box">
        <strong>Training Data for FinBERT</strong><br/><br/>
        <strong>Pre-training corpus:</strong><br/>
        - Reuters TRC2 financial news corpus (46,143 articles)<br/>
        - Financial communication texts, analyst reports<br/><br/>
        <strong>Fine-tuning dataset:</strong> Financial PhraseBank<br/>
        - 4,840 sentences from English-language financial news<br/>
        - Labeled by 16 domain-expert annotators<br/>
        - 3 classes: positive, negative, neutral<br/>
        - Agreement levels: 50%, 66%, 75%, 100%<br/>
        - Created at Aalto University (Malo et al., 2014)<br/>
        - HuggingFace: <code>takala/financial_phrasebank</code>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="green-box">
        <strong>FinQA-7B -- Financial Numerical Reasoning</strong><br/><br/>
        <strong>Base model:</strong> Llama2-7B (Meta, 7B parameters)<br/>
        <strong>Fine-tuned by:</strong> Community (truocpham)<br/>
        <strong>HuggingFace:</strong> <code>truocpham/FinQA-7B-Instruct-v0.1</code><br/><br/>
        <strong>What was changed:</strong> Llama2-7B was fine-tuned using QLoRA
        (4-bit quantized Low-Rank Adaptation) on the FinQA dataset to learn
        multi-step numerical reasoning over financial tables.<br/><br/>
        <strong>Apples-to-apples:</strong> In the live demo, we compare
        FinQA-7B (Llama2-7B fine-tuned) against Llama2-7B base and
        Llama2-7B + RAG -- same architecture, same parameter count,
        different approach.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="blue-box">
        <strong>Training Data for FinQA-7B</strong><br/><br/>
        <strong>Dataset:</strong> FinQA (IBM Research, 2021)<br/>
        - 8,281 question-answer pairs<br/>
        - Sourced from S&P 500 company SEC filings (10-K, 10-Q)<br/>
        - Each example contains:<br/>
        &nbsp;&nbsp;- A financial data table<br/>
        &nbsp;&nbsp;- Pre/post context paragraphs<br/>
        &nbsp;&nbsp;- A numerical reasoning question<br/>
        &nbsp;&nbsp;- A step-by-step reasoning program<br/>
        &nbsp;&nbsp;- The correct numerical answer<br/>
        - HuggingFace: <code>ibm-research/finqa</code><br/><br/>
        <strong>Paper:</strong> <em>"FinQA: A Dataset of Numerical Reasoning
        over Financial Data"</em> (Chen et al., EMNLP 2021)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class="green-box">
        <strong>DistilBERT Spam Detector -- Spam / Phishing Classification</strong><br/><br/>
        <strong>Base model:</strong> DistilBERT-base-uncased (66M parameters)<br/>
        <strong>Fine-tuned on:</strong> Phishing &amp; spam email datasets<br/><br/>
        <strong>What was changed:</strong> A binary classification head was added
        and the model was fine-tuned to distinguish spam/phishing emails from
        legitimate (ham) messages. The model learned urgency language, suspicious
        URLs, verification requests, and prize-claim patterns.<br/><br/>
        <strong>Why DistilBERT:</strong> 40% smaller and 60% faster than BERT-base
        while retaining 97% of BERT's language understanding -- ideal for
        high-throughput email filtering.
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="blue-box">
        <strong>Training Data for Spam Detector</strong><br/><br/>
        <strong>Dataset:</strong> Curated phishing &amp; spam corpus<br/>
        - Phishing emails with urgency language and fake verification links<br/>
        - Nigerian prince / lottery scam messages<br/>
        - Get-rich-quick and work-from-home spam<br/>
        - Legitimate business, notification, and newsletter emails<br/><br/>
        <strong>Categories covered:</strong><br/>
        - Phishing (account compromise, verification)<br/>
        - Obvious spam (lottery, prizes, money offers)<br/>
        - Scams (advance-fee fraud)<br/>
        - Ham (business, notifications, newsletters)<br/><br/>
        <strong>Binary classification:</strong> spam vs ham
        </div>
        """, unsafe_allow_html=True)


def slide_training_data_detail():
    """Training data examples and what the models learned"""
    st.markdown('<p class="slide-title">What the Training Data Looks Like</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">Concrete examples from FinQA and Financial PhraseBank</p>', unsafe_allow_html=True)

    st.markdown("### FinQA Training Example (Numerical Reasoning)")
    st.markdown("""
    Each training example teaches the model HOW to reason, not just WHAT to answer:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Input (table + question):**")
        st.code("""Table:
| Segment       | 2019    | 2018    |
|---------------|---------|---------|
| Products      | $4,231  | $3,891  |
| Services      | $2,107  | $1,988  |

Context: "Revenue growth was driven by
increased demand in the Products segment."

Question: "What was the total revenue
growth rate from 2018 to 2019?" """, language=None)

    with col2:
        st.markdown("**Training label (reasoning + answer):**")
        st.code("""Reasoning program:
  add(3891, 1988) -> 5879    [2018 total]
  add(4231, 2107) -> 6338    [2019 total]
  subtract(6338, 5879) -> 459
  divide(459, 5879) -> 0.0781

Answer: 7.81%

The model learns the STEPS, not just
the final number.""", language=None)

    st.markdown("---")
    st.markdown("### Financial PhraseBank Training Examples (Sentiment)")
    st.markdown("The model learns that financial vocabulary has domain-specific meaning:")

    import pandas as pd
    examples = {
        "Sentence": [
            "Operating profit rose to EUR 13.1 million from EUR 8.7 million",
            "Management expects headwinds from deposit competition to persist",
            "The company maintained its quarterly dividend at $0.50",
            "Restructuring charges totaled $450M in one-time write-downs",
            "The Board of Directors will propose a dividend of EUR 0.12 per share",
        ],
        "Label": ["POSITIVE", "NEGATIVE", "NEUTRAL", "NEGATIVE", "NEUTRAL"],
        "Why (what FinBERT learned)": [
            "'Rose' + specific growth numbers = positive",
            "'Headwinds' + 'persist' = negative (domain jargon)",
            "'Maintained' = no change = neutral (not positive!)",
            "'Restructuring charges' + 'write-downs' = negative",
            "Dividend proposal = routine announcement = neutral",
        ],
    }
    st.table(pd.DataFrame(examples))

    st.markdown("""
    <div class="orange-box">
    <strong>Key insight:</strong> The training data teaches the model that "headwinds"
    is negative and "maintained" is neutral -- vocabulary meanings that are specific
    to finance. A generic model without this training would not know these distinctions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Spam Detection Training Examples")
    st.markdown("The fine-tuned DistilBERT learns to recognise phishing patterns that surface-level similarity misses:")

    spam_examples = {
        "Email Text": [
            "URGENT: Your account has been compromised. Click to verify.",
            "Q4 Board Meeting Agenda -- Hi team, attached is the agenda...",
            "Your prescription is ready for pickup at Walgreens on Main St.",
            "Make $5000 per day from home! No experience needed.",
            "Verify your PayPal account within 24 hours to avoid suspension.",
        ],
        "Label": ["SPAM", "HAM", "HAM", "SPAM", "SPAM"],
        "Why (what the model learned)": [
            "Urgency + 'compromised' + 'verify' = classic phishing pattern",
            "Business context, named event, team address = legitimate",
            "Pharmacy notification -- no urgency, no action demanded",
            "Exaggerated money claims + 'no experience' = get-rich-quick scam",
            "Brand impersonation + deadline threat + 'verify' = phishing",
        ],
    }
    st.table(pd.DataFrame(spam_examples))

    st.markdown("""
    <div class="orange-box">
    <strong>Key insight:</strong> RAG retrieves similar-looking emails but can confuse
    a legitimate pharmacy notification with medication spam, or a real shipping notice
    with a phishing link.  Fine-tuning embeds the <em>patterns</em> (urgency + verification
    + deadline = phishing) directly into the model weights.
    </div>
    """, unsafe_allow_html=True)


def slide_head_to_head():
    """Head-to-Head Comparison"""
    st.markdown('<p class="slide-title">Head-to-Head: RAG vs Fine-Tuning</p>', unsafe_allow_html=True)

    import pandas as pd
    comparison = {
        "Dimension": [
            "Knowledge Type",
            "Teaches New Skills",
            "Setup Cost",
            "Maintenance",
            "Inference Latency",
            "Accuracy (Domain Tasks)",
            "Output Consistency",
            "Needs Training Data",
            "Source Citations",
            "Handles Fresh Data",
        ],
        "RAG": [
            "External (retrieved)",
            "No",
            "Low ($)",
            "Ongoing (update docs)",
            "Higher (~800ms)",
            "Lower (15-40%)",
            "Variable (65%)",
            "No",
            "Yes",
            "Yes",
        ],
        "Fine-Tuning": [
            "Internal (learned)",
            "Yes",
            "Higher ($$$)",
            "Periodic (retrain)",
            "Lower (~200ms)",
            "Higher (60-95%)",
            "High (98%)",
            "Yes",
            "No",
            "No (static)",
        ],
        "Hybrid": [
            "Both",
            "Yes",
            "Highest",
            "Both",
            "Medium (~450ms)",
            "Highest (65%+)",
            "High (95%)",
            "Yes",
            "Yes",
            "Yes",
        ],
    }

    df = pd.DataFrame(comparison)
    st.table(df)

    st.markdown("""
    > **Key insight:** RAG and Fine-Tuning are **complementary**, not competing approaches.
    > The best solution often combines both.
    """)


def slide_rag_falls_short():
    """Slide 12: When RAG Falls Short"""
    st.markdown('<p class="slide-title">When RAG Falls Short: Real Examples</p>', unsafe_allow_html=True)

    st.markdown("### Example 1: Financial Calculation")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Task:** Calculate revenue growth from a financial table

        **RAG Response:**
        > "Retrieved: Revenue growth formula = (Current - Prior) / Prior..."
        > "The revenue appears to have grown. Consumer Banking
        > increased while Investment Banking declined."

        *Retrieved the formula but couldn't compute the answer.*
        """)
        st.error("Result: Vague, no actual calculation")
    with col2:
        st.markdown("""
        **Fine-Tuned Response:**
        ```
        Step 1: 2022 total = $25,330M
        Step 2: 2023 total = $25,950M
        Step 3: Growth = (25,950-25,330)/25,330
                       = 2.45%
        ```

        *Performed the multi-step calculation correctly.*
        """)
        st.success("Result: Precise answer with reasoning")

    st.markdown("---")

    st.markdown("### Example 2: Sentiment Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Text:** *"Management expects headwinds from deposit competition to persist"*

        **RAG:** "The text contains the word 'competition' and 'persist'.
        Based on retrieved sentiment definitions... **unclear/neutral**"
        """)
        st.error("Missed the nuance of 'headwinds'")
    with col2:
        st.markdown("""
        **FinBERT (fine-tuned):**
        **NEGATIVE** (confidence: 91%)

        Correctly identifies "headwinds" and "persist" as
        negative financial language.
        """)
        st.success("Understands domain-specific vocabulary")

    st.markdown("---")

    st.markdown("### Example 3: Spam / Phishing Detection")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Email:** *"Your prescription is ready for pickup at Walgreens on Main St."*

        **RAG:** Retrieved medication-spam examples with similar vocabulary
        (pills, pharmacy, prescription) and voted **SPAM**.
        """)
        st.error("Misclassified -- legitimate notification confused with med-spam")
    with col2:
        st.markdown("""
        **Fine-Tuned DistilBERT:**
        **HAM** (confidence: 99%)

        The model learned that pharmacy pickup notifications lack
        urgency cues, suspicious links, and verification requests
        that define real spam.
        """)
        st.success("Correctly identified as legitimate")


def slide_decision_framework():
    """Slide 13: Decision Framework"""
    st.markdown('<p class="slide-title">Decision Framework: When to Use What</p>', unsafe_allow_html=True)

    render_mermaid("""
    graph TD
        A["Does the task require<br/><b>NEW REASONING SKILLS?</b>"] -->|YES| B["Does it need<br/><b>FRESH / DYNAMIC data?</b>"]
        A -->|NO| C["Does it need<br/><b>FRESH / DYNAMIC data?</b>"]
        B -->|YES| D["<b>HYBRID</b><br/>Fine-Tune + RAG"]
        B -->|NO| E["<b>FINE-TUNE</b><br/>Best accuracy"]
        C -->|YES| F["<b>RAG</b><br/>Dynamic knowledge"]
        C -->|NO| G["<b>PROMPT ENG.</b><br/>Quick start"]

        style D fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
        style E fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
        style F fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#bf360c
        style G fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
        style A fill:#fafafa,stroke:#424242,stroke-width:2px,color:#212121
        style B fill:#fafafa,stroke:#424242,stroke-width:1px,color:#212121
        style C fill:#fafafa,stroke:#424242,stroke-width:1px,color:#212121
    """, height=420)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="blue-box">
        <strong>Choose RAG when:</strong>
        <ul>
            <li>Data changes frequently</li>
            <li>Need source citations</li>
            <li>No training data available</li>
            <li>Quick deployment needed</li>
            <li>Broad knowledge required</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="green-box">
        <strong>Choose Fine-Tuning when:</strong>
        <ul>
            <li>Task needs specialized skills</li>
            <li>High accuracy is critical</li>
            <li>Consistent output required</li>
            <li>Low latency matters</li>
            <li>Domain-specific reasoning</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="highlight-box">
        <strong>Choose Hybrid when:</strong>
        <ul>
            <li>Maximum accuracy needed</li>
            <li>Complex domain analysis</li>
            <li>Both skills AND fresh data</li>
            <li>Production systems</li>
            <li>Budget allows</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


def slide_finetuning_tools():
    """Slide 14: Fine-Tuning Tools & Platforms"""
    st.markdown('<p class="slide-title">Fine-Tuning: Tools & Platforms</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">The ecosystem for training specialized models</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Open-Source Frameworks")
        st.markdown("""
        <div class="tool-card">
        <strong>Hugging Face Transformers + PEFT</strong><br/>
        The standard library for fine-tuning. PEFT adds LoRA/QLoRA support.<br/>
        <em>Best for: Full control, custom training loops</em><br/>
        <code>pip install transformers peft</code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Unsloth</strong><br/>
        2-5x faster fine-tuning with 70% less memory. Optimized kernels.<br/>
        <em>Best for: Fast iteration, consumer GPUs (even free Colab)</em><br/>
        <code>pip install unsloth</code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Axolotl</strong><br/>
        Config-driven fine-tuning. Define everything in a YAML file.<br/>
        <em>Best for: Reproducible training, multiple model support</em><br/>
        <code>pip install axolotl</code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>LLaMA-Factory</strong><br/>
        Web UI for fine-tuning. Supports 100+ LLMs. Visual training dashboard.<br/>
        <em>Best for: Non-technical users, rapid prototyping</em><br/>
        <code>pip install llamafactory</code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>TRL (Transformer RL)</strong><br/>
        SFT, DPO, and RLHF trainers. Integrates with PEFT for alignment tuning.<br/>
        <em>Best for: Alignment tuning, RLHF, preference learning</em><br/>
        <code>pip install trl</code>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Cloud Platforms & APIs")
        st.markdown("""
        <div class="tool-card">
        <strong>OpenAI Fine-Tuning API</strong><br/>
        Upload data, fine-tune GPT-4o-mini or GPT-4o. Fully managed.<br/>
        <em>Best for: Teams already using OpenAI, minimal setup</em><br/>
        <code>openai api fine_tunes.create -t data.jsonl</code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>AWS SageMaker / Bedrock</strong><br/>
        Enterprise fine-tuning with SageMaker JumpStart or Bedrock custom models.<br/>
        <em>Best for: Enterprise teams, AWS ecosystem integration</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Google Vertex AI</strong><br/>
        Fine-tune Gemini models or open-source models on Google Cloud.<br/>
        <em>Best for: Google Cloud users, Gemini ecosystem</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Together AI / Anyscale</strong><br/>
        Serverless fine-tuning platforms. Pay per training hour.<br/>
        <em>Best for: Cost-effective, no infrastructure management</em>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Data, Eval & Monitoring")
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.markdown("""
        <div class="tool-card">
        <strong>Weights & Biases (W&B)</strong><br/>
        Experiment tracking, run comparison, model registry.
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown("""
        <div class="tool-card">
        <strong>MLflow</strong><br/>
        Open-source ML lifecycle. Model versioning, deployment.
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown("""
        <div class="tool-card">
        <strong>Argilla / Label Studio</strong><br/>
        Data labeling and annotation. Human-in-the-loop for training datasets.
        </div>
        """, unsafe_allow_html=True)
    with mcol4:
        st.markdown("""
        <div class="tool-card">
        <strong>LM Eval Harness</strong><br/>
        Standard benchmark suite for LLMs. Measure fine-tuning impact rigorously.
        </div>
        """, unsafe_allow_html=True)
    with mcol5:
        st.markdown("""
        <div class="tool-card">
        <strong>Ollama / vLLM / TGI</strong><br/>
        Local inference servers. GGUF, AWQ, GPTQ quantization for deployment.
        </div>
        """, unsafe_allow_html=True)


def slide_finetune_local():
    """Slide: How to Fine-Tune - Local Setup"""
    st.markdown('<p class="slide-title">How to Fine-Tune: Local Setup</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">Step-by-step with Unsloth + QLoRA on a single GPU</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="blue-box">
        <strong>Prerequisites</strong><br/>
        - NVIDIA GPU with 8+ GB VRAM (RTX 3060/4060 or better)<br/>
        - CUDA 11.8+ and Python 3.10+<br/>
        - 20-50 GB free disk space for model weights<br/>
        <code>pip install unsloth transformers peft trl datasets</code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Python example: Unsloth + QLoRA workflow**")
        st.code("""# 1. Load base model in 4-bit (QLoRA)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    max_seq_length=2048,
)

# 2. Add LoRA adapters (only ~1-5% of params trained)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
)

# 3. Train with SFTTrainer
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model, train_dataset=dataset,
    args=TrainingArguments(
        num_train_epochs=3, per_device_train_batch_size=4,
        output_dir="./output", learning_rate=2e-4,
    ),
)
trainer.train()

# 4. Save & merge adapter into base model
model.save_pretrained_merged("./my-finetuned-model")""", language="python")

    with col2:
        st.markdown("""
        <div class="green-box">
        <strong>What Happens During Training</strong><br/>
        - Base model weights are <strong>FROZEN</strong> (unchanged)<br/>
        - Small adapter matrices (~50-200 MB) are trained<br/>
        - Training takes 30 min - 4 hours for a 7B model<br/>
        - GPU memory: ~6 GB (QLoRA) vs ~28 GB (full FT)<br/>
        - Result: base model + adapter = specialized model
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Training Data Format (JSONL)**")
        st.code("""{"instruction": "Calculate revenue growth",
 "input": "2022: $500M, 2023: $580M",
 "output": "Growth = (580-500)/500 = 16%"}

# Typically 1,000 - 10,000 examples needed.""", language="json")

        st.markdown("""
        <div class="orange-box">
        <strong>Tips</strong><br/>
        - Start with QLoRA + Unsloth for fastest iteration<br/>
        - Use Llama-2/3 7B or Mistral 7B as base model<br/>
        - Evaluate on held-out test set after each epoch<br/>
        - Export to GGUF for Ollama / llama.cpp deployment
        </div>
        """, unsafe_allow_html=True)


def slide_finetune_aws():
    """Slide: How to Fine-Tune - AWS"""
    st.markdown('<p class="slide-title">How to Fine-Tune: AWS</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">SageMaker JumpStart + Bedrock Custom Models</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Option A: Amazon SageMaker JumpStart")
        st.code("""# 1. Select foundation model from JumpStart hub
#    (Llama 2/3, Mistral, Falcon, etc.)

import sagemaker
from sagemaker.jumpstart.estimator import JumpStartEstimator

estimator = JumpStartEstimator(
    model_id="meta-textgeneration-llama-2-7b",
    instance_type="ml.g5.2xlarge",  # 1x A10G 24GB
    environment={
        "instruction_tuned": "True",
        "epoch": "3",
        "per_device_train_batch_size": "4",
        "lora_r": "16",
    },
)

# 2. Point to training data in S3
estimator.fit({
    "training": "s3://my-bucket/training-data/"
})

# 3. Deploy endpoint
predictor = estimator.deploy(
    instance_type="ml.g5.xlarge"
)""", language="python")

    with col2:
        st.markdown("### Option B: Bedrock Custom Models (No-Code)")
        st.markdown("""
        <div class="blue-box">
        <strong>Fully Managed — 6 Steps</strong><br/>
        1. Upload training data (JSONL) to S3<br/>
        2. Console: Bedrock > Custom models > Create<br/>
        3. Select base model (Llama, Titan, Cohere, etc.)<br/>
        4. Configure hyperparameters (epochs, lr, batch)<br/>
        5. Submit job — AWS handles GPU provisioning<br/>
        6. Deploy as Provisioned Throughput endpoint<br/><br/>
        No ML infrastructure to manage.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>SageMaker vs Bedrock</strong><br/>
        <strong>SageMaker:</strong> Full control, custom code, any model, BYO container<br/>
        <strong>Bedrock:</strong> Managed, simpler, limited to supported models<br/>
        <strong>Both:</strong> LoRA/QLoRA, S3 data, IAM security, VPC isolation
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="orange-box">
        <strong>Typical AWS Costs</strong><br/>
        Training: $2–50/hr (ml.g5.2xlarge ~$5/hr).<br/>
        A 7B QLoRA job: ~$5–25 total.<br/>
        Inference: ml.g5.xlarge ~$1.50/hr, or Bedrock per-token pricing.
        </div>
        """, unsafe_allow_html=True)


def slide_rag_tools():
    """Slide 15: RAG Tools & Infrastructure"""
    st.markdown('<p class="slide-title">RAG: Tools & Infrastructure</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">Building blocks for retrieval-augmented systems</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Vector Databases")
        st.markdown("""
        <div class="tool-card">
        <strong>ChromaDB</strong><br/>
        Open-source, embedded vector DB. Great for prototyping.<br/>
        <em>Best for: Local development, small-medium scale</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Pinecone</strong><br/>
        Fully managed, serverless vector database. Enterprise-ready.<br/>
        <em>Best for: Production, large-scale, zero-ops</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Weaviate</strong><br/>
        Open-source vector DB with hybrid search (vector + keyword).<br/>
        <em>Best for: Complex queries, multi-modal search</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Qdrant / Milvus / pgvector</strong><br/>
        Other popular options. pgvector integrates with PostgreSQL.<br/>
        <em>Best for: Existing PostgreSQL users (pgvector), high-performance (Milvus)</em>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Orchestration Frameworks")
        st.markdown("""
        <div class="tool-card">
        <strong>LangChain</strong><br/>
        Most popular framework for building RAG pipelines. Extensive integrations.<br/>
        <em>Best for: Rapid prototyping, wide ecosystem support</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>LlamaIndex</strong><br/>
        Specialized for data ingestion and indexing. Advanced retrieval strategies.<br/>
        <em>Best for: Complex document structures, advanced RAG patterns</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Haystack (deepset)</strong><br/>
        Production-ready NLP framework. Pipeline-based architecture.<br/>
        <em>Best for: Enterprise NLP, modular pipeline design</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Embedding Models")
        st.markdown("""
        <div class="tool-card">
        <strong>Sentence Transformers / OpenAI Embeddings / Cohere Embed</strong><br/>
        Convert text to vectors for retrieval. Choice affects search quality.<br/>
        <em>Tip: Use domain-specific embeddings for better retrieval accuracy</em>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Embeddings, Eval & Cloud RAG")
    ecol1, ecol2, ecol3, ecol4, ecol5 = st.columns(5)
    with ecol1:
        st.markdown("""
        <div class="tool-card">
        <strong>sentence-transformers</strong><br/>
        Open-source embeddings (MiniLM, E5, BGE). Free, local, privacy-preserving.
        </div>
        """, unsafe_allow_html=True)
    with ecol2:
        st.markdown("""
        <div class="tool-card">
        <strong>OpenAI / Cohere / Voyage Embeddings</strong><br/>
        API-based, high quality. text-embedding-3-large and similar. Best quality at a cost.
        </div>
        """, unsafe_allow_html=True)
    with ecol3:
        st.markdown("""
        <div class="tool-card">
        <strong>RAGAS</strong><br/>
        Evaluate RAG pipelines: faithfulness, relevancy, context recall. Measure RAG quality.
        </div>
        """, unsafe_allow_html=True)
    with ecol4:
        st.markdown("""
        <div class="tool-card">
        <strong>AWS Bedrock Knowledge Bases</strong><br/>
        Managed RAG. Auto-chunking, OpenSearch integration, Bedrock LLMs. Zero infrastructure.
        </div>
        """, unsafe_allow_html=True)
    with ecol5:
        st.markdown("""
        <div class="tool-card">
        <strong>Azure AI Search + OpenAI</strong><br/>
        Enterprise search + GPT. Hybrid retrieval, RBAC. Best for Azure/Microsoft ecosystem.
        </div>
        """, unsafe_allow_html=True)


def slide_rag_local():
    """Slide: How to Build RAG - Local Setup"""
    st.markdown('<p class="slide-title">How to Build RAG: Local Setup</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">Python + ChromaDB + sentence-transformers + any LLM</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("**Prerequisites**")
        st.markdown("""
        <div class="blue-box">
        - Python 3.10+, no GPU required for retrieval<br/>
        - GPU optional (for local LLM via Ollama/vLLM)<br/>
        - Or use API: OpenAI / Anthropic / Groq for generation<br/>
        - Storage: ~1 GB per 100K document chunks<br/>
        <code>pip install chromadb sentence-transformers langchain</code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Python code: ChromaDB + sentence-transformers + LangChain**")
        st.code("""# 1. Load & chunk your documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# 2. Create embeddings & vector store
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # 384-dim
)
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./db"
)

# 3. Query: embed question -> retrieve -> generate
results = vectorstore.similarity_search(
    "What was revenue growth?", k=5
)
context = "\\n".join(r.page_content for r in results)
answer = llm(f"{context}\\n\\nQ: {question}")""", language="python")

    with col2:
        st.markdown("""
        <div class="green-box">
        <strong>Architecture</strong><br/>
        Documents → Chunking (500 chars) → Embedding<br/>
        → ChromaDB (cosine similarity) → Top-K retrieval<br/>
        → LLM prompt = question + retrieved chunks<br/>
        → Answer with source references
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="orange-box">
        <strong>Tuning Tips</strong><br/>
        - <strong>Chunk size:</strong> 300–1000 chars — test what works best<br/>
        - <strong>Overlap:</strong> 10–20% of chunk size avoids split entities<br/>
        - <strong>Top-K:</strong> 3–5 docs balances context vs noise<br/>
        - <strong>Embedding model:</strong> try domain-specific ones for better accuracy<br/>
        - <strong>Re-ranking:</strong> cross-encoder boosts precision significantly
        </div>
        """, unsafe_allow_html=True)


def slide_rag_aws():
    """Slide: How to Build RAG - AWS"""
    st.markdown('<p class="slide-title">How to Build RAG: AWS</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">Bedrock Knowledge Bases + OpenSearch Serverless</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Option A: Amazon Bedrock Knowledge Bases (Managed)")
        st.markdown("""
        <div class="blue-box">
        <strong>Fully Managed RAG in 5 Steps</strong><br/>
        1. Create S3 bucket with your documents<br/>
        &nbsp;&nbsp;&nbsp;(PDF, TXT, CSV, HTML, DOCX supported)<br/><br/>
        2. Console: Bedrock > Knowledge bases > Create<br/>
        &nbsp;&nbsp;&nbsp;- Select embedding model (Titan, Cohere)<br/>
        &nbsp;&nbsp;&nbsp;- Select vector store (auto-creates OpenSearch)<br/><br/>
        3. Sync data source (automatic chunking & embedding)<br/><br/>
        4. Query via RetrieveAndGenerate API
        </div>
        """, unsafe_allow_html=True)

        st.code("""response = bedrock.retrieve_and_generate(
    input={'text': 'What was revenue growth?'},
    knowledgeBaseId='KB_ID',
    modelArn='anthropic.claude-3-sonnet'
)
# Response includes answer + source citations""", language="python")

    with col2:
        st.markdown("### Option B: Custom RAG on AWS")
        st.markdown("""
        <div class="green-box">
        <strong>Build Your Own with AWS Services</strong><br/>
        <strong>Embedding:</strong> Bedrock Embeddings API or SageMaker<br/>
        <strong>Vector DB:</strong> OpenSearch Serverless (vector engine)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or Amazon Aurora pgvector<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or self-hosted Qdrant on ECS/EKS<br/>
        <strong>Orchestration:</strong> Lambda + Step Functions<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or LangChain on ECS/Fargate<br/>
        <strong>Generation:</strong> Bedrock (Claude, Llama, Titan)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or SageMaker endpoint
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Managed vs Custom</strong><br/>
        <strong>Bedrock KB:</strong> Zero infra, auto-sync, simple API, fast start<br/>
        <strong>Custom:</strong> Full control, any vector DB, custom chunking,
        advanced retrieval (hybrid search, re-ranking, filters)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="orange-box">
        <strong>Typical AWS Costs</strong><br/>
        Bedrock KB: Embedding ~$0.10/1M tokens, generation per-token,
        OpenSearch from $0.24/hr (serverless OCU)<br/>
        Storage: S3 ~$0.023/GB/month<br/>
        Estimated total for small RAG system: <strong>$50–200/month</strong>
        </div>
        """, unsafe_allow_html=True)


def slide_data_preparation_tools():
    """Slide 16: Data Preparation & Evaluation Tools"""
    st.markdown('<p class="slide-title">Data Preparation & Evaluation Tools</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Data Preparation")
        st.markdown("""
        <div class="tool-card">
        <strong>Argilla</strong><br/>
        Open-source data labeling for LLMs. Human-in-the-loop annotation.<br/>
        <em>Create high-quality instruction/response pairs for fine-tuning</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Label Studio</strong><br/>
        Multi-purpose data labeling tool. Supports text, image, audio.<br/>
        <em>Flexible labeling workflows for diverse training data needs</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Hugging Face Datasets</strong><br/>
        Library + hub for datasets. 100K+ public datasets available.<br/>
        <em>Find existing domain datasets or share your own</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Synthetic Data Generation</strong><br/>
        Use strong LLMs (GPT-4, Claude) to generate training data.<br/>
        <em>Bootstrap fine-tuning datasets from existing documents</em>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Evaluation & Benchmarking")
        st.markdown("""
        <div class="tool-card">
        <strong>LM Evaluation Harness (EleutherAI)</strong><br/>
        Standard framework for evaluating LLMs on benchmarks.<br/>
        <em>Measure performance before and after fine-tuning</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>RAGAS</strong><br/>
        Framework for evaluating RAG pipelines. Measures faithfulness, relevancy.<br/>
        <em>Answer: "Is my RAG pipeline actually helping?"</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>DeepEval</strong><br/>
        Unit testing framework for LLMs. Test hallucination, toxicity, bias.<br/>
        <em>CI/CD integration for LLM quality assurance</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tool-card">
        <strong>Phoenix (Arize AI)</strong><br/>
        LLM observability and evaluation. Trace, evaluate, and debug in production.<br/>
        <em>Monitor fine-tuned model performance over time</em>
        </div>
        """, unsafe_allow_html=True)


def slide_use_cases():
    """Slide 17: Real-World Use Cases"""
    st.markdown('<p class="slide-title">Real-World Use Cases</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Financial Services
        - **Risk Assessment:** Fine-tuned models analyze risk factors with domain accuracy
        - **Compliance Checking:** Trained to identify regulatory violations
        - **Financial Analysis:** Calculate ratios, interpret tables, compare metrics
        - **Sentiment Analysis:** FinBERT understands "headwinds" = negative

        ### Healthcare
        - **Clinical Notes:** Extract diagnoses, medications, procedures
        - **Medical Coding:** ICD-10 code assignment from clinical text
        - **Drug Interactions:** Fine-tuned on pharmacological databases
        """)

    with col2:
        st.markdown("""
        ### Legal
        - **Contract Review:** Identify clauses, obligations, risks
        - **Case Law Research:** Understand legal precedent and citations
        - **Due Diligence:** Extract key terms from thousands of documents

        ### Cybersecurity & Email Filtering
        - **Spam/Phishing Detection:** Fine-tuned DistilBERT catches urgency + verification patterns
        - **Threat Classification:** Learned phishing vs scam vs legitimate at 95% accuracy
        - **Email Triage:** High-throughput classification with calibrated confidence scores
        """)

    st.info("""
    **Common pattern:** Start with RAG for quick wins, then add fine-tuning
    where accuracy gaps appear. The data from RAG usage often becomes the
    training data for fine-tuning.
    """)


def slide_hybrid():
    """Slide 18: The Hybrid Approach"""
    st.markdown('<p class="slide-title">The Hybrid Approach: Best of Both Worlds</p>', unsafe_allow_html=True)

    render_mermaid("""
    graph TD
        A["User Question + Financial Table"] --> B["Embedding Model<br/><i>all-MiniLM-L6-v2</i>"]
        A --> C["Primary Context<br/><i>Table + Text</i>"]
        B --> D["Vector Store<br/><i>ChromaDB</i>"]
        D --> E["Retrieved Documents<br/><i>Top-K similar chunks</i>"]
        E --> F["<b>FinQA-7B-Instruct</b><br/>Fine-tuned model"]
        C --> F
        F --> G["<b>Answer</b><br/>Domain reasoning + Fresh context + Citations"]

        style A fill:#f5f5f5,stroke:#424242,stroke-width:2px,color:#212121
        style D fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,color:#0d47a1
        style F fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
        style G fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#5d4037
        style B fill:#fafafa,stroke:#9e9e9e,color:#424242
        style C fill:#fafafa,stroke:#9e9e9e,color:#424242
        style E fill:#fafafa,stroke:#9e9e9e,color:#424242
    """, height=550)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### What Fine-Tuning Contributes
        - Numerical reasoning ability
        - Domain-specific patterns
        - Consistent output format
        - Lower error rate
        """)
    with col2:
        st.markdown("""
        ### What RAG Contributes
        - Fresh, updatable knowledge
        - Source citations for audit
        - Broader context coverage
        - Reduced hallucination
        """)

    st.success("""
    **Result:** The hybrid approach achieves **65.8% accuracy** on FinQA, compared to
    61.2% for fine-tuning alone and 15.3% for RAG alone.
    **Sentiment:** 75% hybrid vs 70% FT-only vs 65% RAG-only.
    **Spam Detection:** 95% hybrid = 95% FT-only > 90% RAG-only (with higher confidence).
    """)


def slide_cost_roi():
    """Slide 19: Cost & ROI"""
    st.markdown('<p class="slide-title">Cost & ROI Considerations</p>', unsafe_allow_html=True)

    import pandas as pd
    cost_data = {
        "": ["Setup Cost", "Per-Query Cost", "Maintenance", "Time to Deploy", "Accuracy ROI", "Best For"],
        "Prompt Engineering": ["Free", "API cost only", "Manual prompt updates", "Hours", "Low", "Prototyping"],
        "RAG": ["$100-$1K", "API + retrieval cost", "Document updates", "Days", "Medium", "Knowledge lookup"],
        "Fine-Tuning": ["$500-$50K+", "Lower inference cost", "Periodic retraining", "Days-Weeks", "High", "Specialized tasks"],
        "Hybrid": ["$1K-$100K+", "Medium", "Both", "Weeks", "Highest", "Production systems"],
    }
    st.table(pd.DataFrame(cost_data))

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### The ROI Argument for Fine-Tuning
        - Training cost is a **one-time investment**
        - Lower per-query cost than RAG (no retrieval step)
        - Higher accuracy = fewer errors = less human review
        - Faster responses = better user experience
        - **Break-even:** Often within weeks for high-volume use cases
        """)
    with col2:
        st.markdown("""
        ### When RAG Has Better ROI
        - Low query volume (< 1K queries/day)
        - Rapidly changing knowledge base
        - No training data available
        - Need to launch in < 1 week
        - Proof-of-concept stage
        """)

    st.warning("**Key question:** How much does a wrong answer cost your organization? If the answer is 'a lot,' fine-tuning's accuracy advantage quickly pays for itself.")


def slide_key_takeaways():
    """Slide 20: Key Takeaways"""
    st.markdown('<p class="slide-title">Key Takeaways</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box">
    <h3 style="color: white; margin-top: 0;">The Core Message</h3>
    <p style="font-size: 1.2rem;">
    Fine-tuning teaches a model new <strong>SKILLS</strong>.<br/>
    RAG gives a model new <strong>INFORMATION</strong>.<br/>
    The hybrid approach provides <strong>BOTH</strong>.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    takeaways = [
        ("Fine-tuning is essential for specialized reasoning",
         "When your task requires domain-specific calculations, consistent output formats, or learned reasoning patterns, fine-tuning delivers 61%+ accuracy vs 15% for RAG on FinQA, and 95% vs 90% on spam detection."),
        ("RAG is valuable for dynamic knowledge",
         "When your data changes frequently, you need source citations for audit/compliance, or you need to deploy quickly without training data — RAG is the right first choice."),
        ("Best production systems combine both (hybrid)",
         "The hybrid approach achieves 65.8% on FinQA vs 61.2% FT-only vs 15.3% RAG-only, and matches fine-tuning's 95% on spam detection. Fine-tuning provides the reasoning skills; RAG provides fresh, citable context."),
        ("Modern tools make fine-tuning accessible",
         "Unsloth + QLoRA lets you fine-tune a 7B model on a consumer GPU (8 GB VRAM) in under 4 hours. You no longer need a data center or an ML PhD to get started."),
        ("Start with RAG, add fine-tuning where gaps appear",
         "RAG is faster and cheaper to deploy. Use it first. The queries where RAG fails or gives low-confidence answers become your training data for fine-tuning."),
        ("The cost of fine-tuning is an investment",
         "A QLoRA fine-tuning job on AWS costs $5–25. Higher accuracy means fewer wrong answers, fewer human review cycles, and faster responses. Break-even often within weeks at production volume."),
    ]

    for i, (title, desc) in enumerate(takeaways, 1):
        st.markdown(f"""
        <div class="green-box">
        <strong>{i}. {title}</strong><br/>
        {desc}
        </div>
        """, unsafe_allow_html=True)


def slide_demo_intro():
    """Slide 21: Live Demo Introduction"""
    st.markdown('<p class="slide-title">Live Demo: See the Difference</p>', unsafe_allow_html=True)

    st.markdown("""
    ### What we'll demonstrate:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="tool-card">
        <h3>Demo 1: Sentiment Analysis</h3>
        <p>Compare <strong>FinBERT</strong> (fine-tuned) vs a generic keyword-based approach.</p>
        <p>See the speed and accuracy difference with your own eyes.</p>
        <p><em>Navigate to: Sentiment Analysis page</em></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="tool-card">
        <h3>Demo 2: Financial Reasoning</h3>
        <p>Compare <strong>Fine-Tuned vs RAG vs Base Model</strong> on real financial calculations.</p>
        <p>Watch RAG struggle with math while fine-tuned models compute accurately.</p>
        <p><em>Navigate to: Numerical Reasoning page</em></p>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class="tool-card">
        <h3>Demo 3: Spam Detection</h3>
        <p>Compare <strong>Fine-Tuned DistilBERT</strong> vs <strong>RAG-based</strong> spam classification.</p>
        <p>See how fine-tuning recognises phishing patterns that RAG misses.</p>
        <p><em>Navigate to: Spam Detection page</em></p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="tool-card">
        <h3>Demo 4: Benchmarks at Scale</h3>
        <p>Pre-computed results across <strong>all 4 experiments</strong>.</p>
        <p>See the aggregate numbers that prove fine-tuning's advantage.</p>
        <p><em>Navigate to: Benchmark Results page</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.success("**Use the sidebar navigation** to switch to the demo pages. Let's see fine-tuning in action!")


# ---------------------------------------------------------------------------
# Benchmark Results slides (loaded from data/benchmark_results.json)
# ---------------------------------------------------------------------------
import json
from pathlib import Path as _Path

_RESULTS_PATH = _Path(__file__).parent.parent.parent / "data" / "benchmark_results.json"
_benchmark_data = {}
if _RESULTS_PATH.exists():
    try:
        with open(_RESULTS_PATH) as _f:
            _benchmark_data = json.load(_f)
    except (json.JSONDecodeError, OSError):
        pass

_MODEL_FAMILY_PATH = _Path(__file__).parent.parent.parent / "data" / "model_family_results.json"
_model_family_data = {}
if _MODEL_FAMILY_PATH.exists():
    try:
        with open(_MODEL_FAMILY_PATH) as _f:
            _model_family_data = json.load(_f)
    except (json.JSONDecodeError, OSError):
        pass

_COLORS_MAP = {
    "base": "#E84D4D", "finbert": "#2EA04E", "finetuned": "#2EA04E",
    "rag": "#428BCA", "hybrid": "#F09319",
}


def _render_benchmark_section(section_key, title, labels):
    """Render accuracy, latency, and quality metrics for one benchmark section."""
    import plotly.graph_objects as go
    sections = _benchmark_data.get("sections", {})
    if section_key not in sections:
        st.warning(f"No benchmark data found for {title}. Run the benchmarks first.")
        return
    sec = sections[section_key]
    summary = sec.get("summary", {})
    models = sec.get("models", [])
    arch = sec.get("architecture", "")

    st.markdown(f'<p class="slide-title">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="slide-subtitle">Architecture: {arch}</p>', unsafe_allow_html=True)

    # Accuracy metric cards
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        sm = summary.get(m, {})
        with col:
            st.metric(
                labels.get(m, m),
                f"{sm.get('accuracy', 0)}%",
                delta=f"{sm.get('correct', 0)}/{sm.get('total', 0)}",
            )

    # Accuracy + Latency charts side by side
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig = go.Figure()
        for m in models:
            sm = summary.get(m, {})
            fig.add_trace(go.Bar(
                name=labels.get(m, m).split("(")[0].strip(),
                x=["Accuracy (%)"], y=[sm.get("accuracy", 0)],
                marker_color=_COLORS_MAP.get(m, "#999"),
                text=[f"{sm.get('accuracy', 0)}%"], textposition="auto",
            ))
        fig.update_layout(title="Accuracy Comparison", barmode="group",
                          yaxis_range=[0, 105], height=350, margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True, key=f"pres_acc_{section_key}")

    with chart_col2:
        lat_data = {m: summary.get(m, {}).get("avg_latency_ms") for m in models}
        if any(v for v in lat_data.values()):
            fig2 = go.Figure()
            for m in models:
                v = lat_data.get(m)
                if v:
                    fig2.add_trace(go.Bar(
                        name=labels.get(m, m).split("(")[0].strip(),
                        x=["Avg Latency (ms)"], y=[v],
                        marker_color=_COLORS_MAP.get(m, "#999"),
                        text=[f"{v:.0f}ms"], textposition="auto",
                    ))
            fig2.update_layout(title="Average Latency", barmode="group",
                               height=350, margin=dict(t=40, b=30))
            st.plotly_chart(fig2, use_container_width=True, key=f"pres_lat_{section_key}")

    # Token & Cost metrics
    tok_data = {m: summary.get(m, {}).get("avg_tokens_per_query", 0) for m in models}
    cost_data = {m: summary.get(m, {}).get("cost_per_1k_queries_usd", 0) for m in models}
    if any(tok_data.values()):
        st.markdown("---")
        tok_cols = st.columns(len(models))
        for col, m in zip(tok_cols, models):
            sm = summary.get(m, {})
            with col:
                st.metric(
                    labels.get(m, m).split("(")[0].strip(),
                    f"{sm.get('avg_tokens_per_query', 0):,} tok/query",
                )
                cost = sm.get("cost_per_1k_queries_usd", 0)
                st.caption(f"Cost/1K queries: ${cost:.4f}")
                tps = sm.get("avg_throughput_tps")
                if tps:
                    st.caption(f"Throughput: {tps:,.0f} tok/s")

    # F1 (sentiment) or MAPE (numerical)
    f1_data = {m: summary.get(m, {}).get("f1_macro") for m in models}
    mape_data = {m: summary.get(m, {}).get("mape") for m in models}

    if any(v is not None for v in f1_data.values()):
        st.markdown("---")
        st.markdown("##### F1 Score & Precision / Recall")
        f1_cols = st.columns(len(models))
        for col, m in zip(f1_cols, models):
            sm = summary.get(m, {})
            with col:
                st.metric(labels.get(m, m).split("(")[0].strip(),
                          f"F1: {sm.get('f1_macro', 0):.3f}")
                st.caption(f"P: {sm.get('precision_macro', 0):.3f} / R: {sm.get('recall_macro', 0):.3f}")

    if any(v is not None for v in mape_data.values()):
        st.markdown("---")
        st.markdown("##### Mean Absolute Percentage Error")
        mape_cols = st.columns(len(models))
        for col, m in zip(mape_cols, models):
            sm = summary.get(m, {})
            with col:
                mape = sm.get("mape", 0)
                st.metric(labels.get(m, m).split("(")[0].strip(),
                          f"MAPE: {mape:.1f}%" if mape else "N/A")

    # Category breakdown
    cat_data = {k: v for k, v in summary.items() if k.startswith("category_")}
    if cat_data:
        st.markdown("---")
        import pandas as pd
        cat_rows = []
        for key in sorted(cat_data.keys()):
            val = cat_data[key]
            cat_name = key.replace("category_", "").replace("_", " ").title()
            row = {"Category": cat_name, "Cases": val["total"]}
            for m in models:
                row[labels.get(m, m).split("(")[0].strip()] = f"{val.get(f'{m}_accuracy', 0)}%"
            cat_rows.append(row)
        with st.expander("Category Breakdown"):
            st.table(pd.DataFrame(cat_rows))


def slide_benchmark_overview():
    """Benchmark Experiments Overview"""
    import pandas as pd
    st.markdown('<p class="slide-title">Benchmark Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">Every number measured by running our actual models</p>', unsafe_allow_html=True)

    ts = _benchmark_data.get("timestamp", "not yet run")
    st.caption(f"Results from: {ts}")

    st.table(pd.DataFrame([
        {"Experiment": "Section 1", "Architecture": "BERT-base (110M)", "Approaches": "Base, FinBERT, RAG, Hybrid", "Task": "Sentiment classification"},
        {"Experiment": "Section 2", "Architecture": "Llama2-7B (7B)", "Approaches": "Base, FinQA-7B, RAG, Hybrid", "Task": "Numerical reasoning"},
        {"Experiment": "Section 3", "Architecture": "Llama2-7B (7B)", "Approaches": "Base, FinQA-7B, RAG, Hybrid", "Task": "Financial ratio calculation"},
        {"Experiment": "Section 4", "Architecture": "DistilBERT (66M)", "Approaches": "Base, Fine-tuned, RAG, Hybrid", "Task": "Spam / phishing detection"},
        {"Experiment": "Section 5", "Architecture": "DistilBERT (66M) vs GPT-4o-mini (~8B)", "Approaches": "Two fine-tuned models compared", "Task": "Model size impact on spam detection"},
    ]))

    st.markdown("""
    <div class="blue-box">
    <strong>Methodology:</strong> Each experiment uses the <strong>same architecture and parameter count</strong>.
    The only variable is the approach (base, fine-tuned, RAG, or hybrid).
    All results measured in our environment with our models.
    </div>
    """, unsafe_allow_html=True)


def slide_benchmark_sentiment():
    """Benchmark: BERT 110M Sentiment"""
    _render_benchmark_section(
        "bert_110m_sentiment",
        "Benchmark: BERT 110M -- Sentiment Classification",
        {"base": "Base BERT", "finbert": "FinBERT (fine-tuned)",
         "rag": "BERT + RAG", "hybrid": "FinBERT + RAG"},
    )


def slide_benchmark_numerical():
    """Benchmark: Llama2 7B Numerical Reasoning"""
    _render_benchmark_section(
        "llama2_7b_numerical",
        "Benchmark: Llama2 7B -- Numerical Reasoning",
        {"base": "Base Llama2-7B", "finetuned": "FinQA-7B (fine-tuned)",
         "rag": "Llama2-7B + RAG", "hybrid": "FinQA-7B + RAG"},
    )


def slide_benchmark_ratios():
    """Benchmark: Llama2 7B Financial Ratios"""
    _render_benchmark_section(
        "llama2_7b_financial_ratios",
        "Benchmark: Llama2 7B -- Financial Ratios",
        {"base": "Base Llama2-7B", "finetuned": "FinQA-7B (fine-tuned)",
         "rag": "Llama2-7B + RAG", "hybrid": "FinQA-7B + RAG"},
    )


def slide_benchmark_spam():
    """Benchmark: DistilBERT 66M Spam Detection"""
    _render_benchmark_section(
        "distilbert_66m_spam",
        "Benchmark: DistilBERT 66M -- Spam Detection",
        {"base": "Base DistilBERT", "finetuned": "Fine-tuned (spam-trained)",
         "rag": "DistilBERT + RAG", "hybrid": "Fine-tuned + RAG"},
    )


def slide_benchmark_summary():
    """Benchmark: All Experiments at a Glance"""
    import plotly.graph_objects as go
    st.markdown('<p class="slide-title">All Experiments at a Glance</p>', unsafe_allow_html=True)

    sections = _benchmark_data.get("sections", {})
    if not sections:
        st.warning("No benchmark data available. Run the benchmarks first.")
        return

    section_info = [
        ("bert_110m_sentiment", "Sentiment (BERT 110M)",
         {"base": "Base", "finbert": "FinBERT", "rag": "RAG", "hybrid": "Hybrid"}),
        ("llama2_7b_numerical", "Numerical (Llama2 7B)",
         {"base": "Base", "finetuned": "FinQA-7B", "rag": "RAG", "hybrid": "Hybrid"}),
        ("llama2_7b_financial_ratios", "Fin. Ratios (Llama2 7B)",
         {"base": "Base", "finetuned": "FinQA-7B", "rag": "RAG", "hybrid": "Hybrid"}),
        ("distilbert_66m_spam", "Spam (DistilBERT 66M)",
         {"base": "Base", "finetuned": "Fine-tuned", "rag": "RAG", "hybrid": "Hybrid"}),
    ]

    # Grouped bar chart: accuracy across all experiments
    fig = go.Figure()
    approach_names = ["base", "finetuned", "rag", "hybrid"]
    approach_labels = {"base": "Base", "finetuned": "Fine-Tuned", "rag": "RAG", "hybrid": "Hybrid"}

    for approach in approach_names:
        accs = []
        exp_names = []
        for sec_key, sec_label, _ in section_info:
            sec = sections.get(sec_key, {})
            summary = sec.get("summary", {})
            # For sentiment, "finetuned" maps to "finbert"
            m = approach
            if sec_key == "bert_110m_sentiment" and approach == "finetuned":
                m = "finbert"
            sm = summary.get(m, {})
            accs.append(sm.get("accuracy", 0))
            exp_names.append(sec_label)

        fig.add_trace(go.Bar(
            name=approach_labels.get(approach, approach),
            x=exp_names, y=accs,
            marker_color=_COLORS_MAP.get(approach, "#999"),
            text=[f"{a}%" for a in accs], textposition="auto",
        ))

    fig.update_layout(
        title="Accuracy Across All Experiments",
        barmode="group", yaxis_range=[0, 105],
        height=450, margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, key="pres_summary_all")

    st.markdown("""
    <div class="green-box">
    <strong>Key Insight:</strong> Fine-tuning consistently outperforms base models across all four experiments
    -- from sentiment classification and numerical reasoning to spam detection.
    The hybrid approach (fine-tuning + RAG) provides the best or near-best results in every category.
    </div>
    """, unsafe_allow_html=True)


def slide_benchmark_model_family():
    """Benchmark: Does Model Size Matter for Fine-Tuning?"""
    import plotly.graph_objects as go
    import pandas as pd

    st.markdown('<p class="slide-title">Does Model Size Matter for Fine-Tuning?</p>', unsafe_allow_html=True)
    st.markdown('<p class="slide-subtitle">Fine-tuned DistilBERT (66M) vs Fine-tuned GPT-4o-mini (~8B) on spam detection</p>', unsafe_allow_html=True)

    sections = _model_family_data.get("sections", {})
    if not sections:
        st.warning("No model family benchmark data available. Run `python app/model_family_benchmark.py` first.")
        return

    ts = _model_family_data.get("timestamp", "not yet run")
    st.caption(f"Results from: {ts}")

    models = ["distilbert_ft", "gpt4omini_ft"]
    labels = {
        "distilbert_ft": "Fine-tuned DistilBERT (66M)",
        "gpt4omini_ft": "Fine-tuned GPT-4o-mini (~8B)",
    }
    colors = {"distilbert_ft": "#ff6b35", "gpt4omini_ft": "#4a90d9"}

    # Model comparison cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="border: 2px solid #ff6b35; border-radius: 10px; padding: 12px; text-align: center;">
        <h4 style="color: #ff6b35; margin: 0;">DistilBERT</h4>
        <p style="font-size: 1.8em; font-weight: bold; margin: 0.3em 0;">66M params</p>
        <p style="margin: 0;">Local inference | Near-zero cost</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="border: 2px solid #4a90d9; border-radius: 10px; padding: 12px; text-align: center;">
        <h4 style="color: #4a90d9; margin: 0;">GPT-4o-mini</h4>
        <p style="font-size: 1.8em; font-weight: bold; margin: 0.3em 0;">~8B params</p>
        <p style="margin: 0;">OpenAI API | $0.30/$1.20 per 1M tokens</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Accuracy comparison: Basic vs Adversarial
    basic = sections.get("basic_spam", {})
    adv = sections.get("adversarial_spam", {})
    basic_summary = basic.get("summary", {})
    adv_summary = adv.get("summary", {})

    # Metric cards for both sections
    if basic_summary:
        st.markdown("##### Basic Spam Detection (20 cases)")
        cols = st.columns(len(models))
        for col, m in zip(cols, models):
            s = basic_summary.get(m, {})
            with col:
                st.metric(labels[m], f"{s.get('accuracy', 0)}%",
                          delta=f"{s.get('correct', 0)}/{s.get('total', 0)}")

    if adv_summary:
        st.markdown("##### Adversarial Spam Detection (30 cases)")
        cols = st.columns(len(models))
        for col, m in zip(cols, models):
            s = adv_summary.get(m, {})
            with col:
                st.metric(labels[m], f"{s.get('accuracy', 0)}%",
                          delta=f"{s.get('correct', 0)}/{s.get('total', 0)}")

    # Combined accuracy chart
    fig = go.Figure()
    for m in models:
        accs = []
        x_labels = []
        for sec_key, sec_label in [("basic_spam", "Basic"), ("adversarial_spam", "Adversarial")]:
            sec = sections.get(sec_key, {})
            summary = sec.get("summary", {})
            if summary:
                accs.append(summary.get(m, {}).get("accuracy", 0))
                x_labels.append(sec_label)
        if accs:
            fig.add_trace(go.Bar(
                name=labels[m], x=x_labels, y=accs,
                marker_color=colors.get(m, "#999"),
                text=[f"{a}%" for a in accs], textposition="auto",
            ))
    fig.update_layout(
        title="Accuracy: Basic vs Adversarial",
        barmode="group", yaxis_range=[0, 105],
        height=400, margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, key="pres_mf_acc")

    # Latency and cost comparison
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_lat = go.Figure()
        for m in models:
            lats = []
            x_labels = []
            for sec_key, sec_label in [("basic_spam", "Basic"), ("adversarial_spam", "Adversarial")]:
                sec = sections.get(sec_key, {})
                summary = sec.get("summary", {})
                if summary:
                    lat = summary.get(m, {}).get("avg_latency_ms", 0)
                    lats.append(lat)
                    x_labels.append(sec_label)
            if lats:
                fig_lat.add_trace(go.Bar(
                    name=labels[m].split("(")[0].strip(), x=x_labels, y=lats,
                    marker_color=colors.get(m, "#999"),
                    text=[f"{l:.0f}ms" for l in lats], textposition="auto",
                ))
        fig_lat.update_layout(title="Average Latency", barmode="group",
                              height=350, margin=dict(t=40, b=30))
        st.plotly_chart(fig_lat, use_container_width=True, key="pres_mf_lat")

    with chart_col2:
        fig_cost = go.Figure()
        for m in models:
            costs = []
            x_labels = []
            for sec_key, sec_label in [("basic_spam", "Basic"), ("adversarial_spam", "Adversarial")]:
                sec = sections.get(sec_key, {})
                summary = sec.get("summary", {})
                if summary:
                    cost = summary.get(m, {}).get("cost_per_1k_queries_usd", 0)
                    costs.append(cost)
                    x_labels.append(sec_label)
            if costs:
                fig_cost.add_trace(go.Bar(
                    name=labels[m].split("(")[0].strip(), x=x_labels, y=costs,
                    marker_color=colors.get(m, "#999"),
                    text=[f"${c:.4f}" for c in costs], textposition="auto",
                ))
        fig_cost.update_layout(title="Cost per 1K Queries ($)", barmode="group",
                               height=350, margin=dict(t=40, b=30))
        st.plotly_chart(fig_cost, use_container_width=True, key="pres_mf_cost")

    # Key findings
    findings = []
    if basic_summary and adv_summary:
        db_basic = basic_summary.get("distilbert_ft", {}).get("accuracy", 0)
        gpt_basic = basic_summary.get("gpt4omini_ft", {}).get("accuracy", 0)
        db_adv = adv_summary.get("distilbert_ft", {}).get("accuracy", 0)
        gpt_adv = adv_summary.get("gpt4omini_ft", {}).get("accuracy", 0)

        if gpt_basic > db_basic:
            findings.append(f"GPT-4o-mini outperforms DistilBERT on basic cases ({gpt_basic}% vs {db_basic}%)")
        elif db_basic > gpt_basic:
            findings.append(f"DistilBERT matches or outperforms GPT-4o-mini on basic cases ({db_basic}% vs {gpt_basic}%)")
        else:
            findings.append(f"Both models tied on basic cases at {db_basic}%")

        if gpt_adv > db_adv:
            findings.append(f"GPT-4o-mini more robust on adversarial cases ({gpt_adv}% vs {db_adv}%)")
        elif db_adv > gpt_adv:
            findings.append(f"DistilBERT more robust on adversarial cases ({db_adv}% vs {gpt_adv}%)")

        db_lat = basic_summary.get("distilbert_ft", {}).get("avg_latency_ms", 0)
        gpt_lat = basic_summary.get("gpt4omini_ft", {}).get("avg_latency_ms", 0)
        if db_lat and gpt_lat:
            speed_ratio = gpt_lat / db_lat
            findings.append(f"DistilBERT is **{speed_ratio:.0f}x faster** ({db_lat:.0f}ms vs {gpt_lat:.0f}ms)")

        findings.append("Size ratio: **~121x** more parameters yields only a **modest accuracy gain**")

    if findings:
        finding_items = "".join(f"<br/>&#8226; {f}" for f in findings)
        st.markdown(f"""
        <div class="highlight-box">
        <strong>Key Findings -- Does 121x More Parameters = Better Spam Detection?</strong>
        {finding_items}
        </div>
        """, unsafe_allow_html=True)

    # LLM-as-Judge results
    if _model_family_data.get("with_judge"):
        judge_sums = _model_family_data.get("judge_summaries", {})
        if judge_sums:
            st.markdown("---")
            st.markdown("##### LLM-as-Judge Evaluation")

            for sec_key, sec_label in [("basic_spam", "Basic"), ("adversarial_spam", "Adversarial")]:
                js = judge_sums.get(sec_key, {})
                if not js:
                    continue
                st.markdown(f"**{sec_label} Cases**")
                judge_cols = st.columns(len(models))
                for col, m in zip(judge_cols, models):
                    jm = js.get(m, {})
                    with col:
                        if jm.get("count", 0) > 0:
                            st.metric(labels[m], f"Overall: {jm['overall']:.1f}/5")
                            st.caption(
                                f"C={jm['correctness']:.1f} | "
                                f"R={jm['reasoning_quality']:.1f} | "
                                f"F={jm['faithfulness']:.1f}"
                            )
                        else:
                            st.metric(labels[m], "N/A")

            # Radar chart
            fig_radar = go.Figure()
            radar_cats = ["Correctness", "Reasoning", "Faithfulness"]
            for sec_key, dash in [("basic_spam", None), ("adversarial_spam", "dash")]:
                js = judge_sums.get(sec_key, {})
                sec_short = "Basic" if sec_key == "basic_spam" else "Adv"
                for m in models:
                    jm = js.get(m, {})
                    if jm.get("count", 0) > 0:
                        vals = [jm["correctness"], jm["reasoning_quality"], jm["faithfulness"]]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals + [vals[0]],
                            theta=radar_cats + [radar_cats[0]],
                            fill="toself",
                            name=f"{labels[m].split('(')[0].strip()} ({sec_short})",
                            line=dict(color=colors.get(m, "#999"),
                                      dash=dash),
                            opacity=0.7 if dash else 1.0,
                        ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                title="Judge Scores Radar", height=400,
            )
            st.plotly_chart(fig_radar, use_container_width=True, key="pres_mf_radar")


# ---------------------------------------------------------------------------
# Slide registry
# ---------------------------------------------------------------------------
SLIDES = [
    ("Title", slide_title),
    ("Agenda", slide_agenda),
    ("What Are LLMs?", slide_what_are_llms),
    ("The Specialization Challenge", slide_specialization_challenge),
    ("Three Approaches", slide_three_approaches),
    ("RAG: How It Works", slide_rag_explained),
    ("RAG: Benefits", slide_rag_benefits),
    ("RAG: Limitations", slide_rag_limitations),
    ("Fine-Tuning: How It Works", slide_finetuning_explained),
    ("Fine-Tuning: Methods", slide_finetuning_methods),
    ("Fine-Tuning: Benefits", slide_finetuning_benefits),
    ("Our Models & Training Data", slide_models_used),
    ("Training Data Examples", slide_training_data_detail),
    ("Head-to-Head Comparison", slide_head_to_head),
    ("When RAG Falls Short", slide_rag_falls_short),
    ("Decision Framework", slide_decision_framework),
    ("Fine-Tuning Tools", slide_finetuning_tools),
    ("Fine-Tune: Local Setup", slide_finetune_local),
    ("Fine-Tune: AWS", slide_finetune_aws),
    ("RAG Tools", slide_rag_tools),
    ("RAG: Local Setup", slide_rag_local),
    ("RAG: AWS", slide_rag_aws),
    ("Data & Evaluation Tools", slide_data_preparation_tools),
    ("Real-World Use Cases", slide_use_cases),
    ("The Hybrid Approach", slide_hybrid),
    ("Cost & ROI", slide_cost_roi),
    ("Key Takeaways", slide_key_takeaways),
    ("Benchmark Overview", slide_benchmark_overview),
    ("Benchmark: Sentiment", slide_benchmark_sentiment),
    ("Benchmark: Numerical", slide_benchmark_numerical),
    ("Benchmark: Financial Ratios", slide_benchmark_ratios),
    ("Benchmark: Spam Detection", slide_benchmark_spam),
    ("Benchmark: Summary", slide_benchmark_summary),
    ("Benchmark: Model Size", slide_benchmark_model_family),
    ("Live Demo", slide_demo_intro),
]

TOTAL_SLIDES = len(SLIDES)

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
if "slide_num" not in st.session_state:
    st.session_state.slide_num = 0

# Top navigation bar
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 4, 1])

with nav_col1:
    if st.button("< Previous", use_container_width=True, disabled=(st.session_state.slide_num == 0)):
        st.session_state.slide_num -= 1
        st.rerun()

with nav_col2:
    if st.button("Next >", use_container_width=True, disabled=(st.session_state.slide_num >= TOTAL_SLIDES - 1)):
        st.session_state.slide_num += 1
        st.rerun()

with nav_col3:
    slide_name = SLIDES[st.session_state.slide_num][0]
    st.progress(
        (st.session_state.slide_num + 1) / TOTAL_SLIDES,
        text=f"Slide {st.session_state.slide_num + 1}/{TOTAL_SLIDES}: {slide_name}",
    )

with nav_col4:
    jump_to = st.selectbox(
        "Jump to:",
        range(TOTAL_SLIDES),
        index=st.session_state.slide_num,
        format_func=lambda i: f"{i+1}. {SLIDES[i][0]}",
        label_visibility="collapsed",
    )
    if jump_to != st.session_state.slide_num:
        st.session_state.slide_num = jump_to
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Render current slide
# ---------------------------------------------------------------------------
SLIDES[st.session_state.slide_num][1]()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()

# Bottom navigation (repeated for convenience)
bot_col1, bot_col2, bot_col3 = st.columns([1, 4, 1])
with bot_col1:
    if st.button("< Prev", use_container_width=True, key="bot_prev",
                 disabled=(st.session_state.slide_num == 0)):
        st.session_state.slide_num -= 1
        st.rerun()
with bot_col2:
    st.caption(f"Slide {st.session_state.slide_num + 1} of {TOTAL_SLIDES}")
with bot_col3:
    if st.button("Next >", use_container_width=True, key="bot_next",
                 disabled=(st.session_state.slide_num >= TOTAL_SLIDES - 1)):
        st.session_state.slide_num += 1
        st.rerun()

# Sidebar
with st.sidebar:
    st.header("Slide Outline")
    for i, (name, _) in enumerate(SLIDES):
        prefix = "**>>** " if i == st.session_state.slide_num else ""
        if st.button(f"{prefix}{i+1}. {name}", key=f"sidebar_{i}", use_container_width=True):
            st.session_state.slide_num = i
            st.rerun()
