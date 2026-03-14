"""
Interactive Presentation Slides
LLM Fine-Tuning: Maximizing AI for Specialized Tasks
Navigate with Previous/Next buttons or keyboard shortcuts.
"""
import streamlit as st

st.set_page_config(
    page_title="Presentation - Fine-Tuning vs RAG",
    page_icon="FT",
    layout="wide",
)

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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .green-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem 1.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .red-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem 1.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .blue-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem 1.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .orange-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem 1.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .tool-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    .comparison-table th {
        background-color: #0066cc;
        color: white;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0066cc;
    }
    .section-label {
        font-size: 0.8rem;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Slide definitions
# ---------------------------------------------------------------------------
TOTAL_SLIDES = 22


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
        13. Benchmark results
        14. Key takeaways & Q&A
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

    st.markdown("""
    ### How RAG Works
    """)

    st.code("""
    User Question
         |
         v
    [1. EMBED] -----> Convert question to vector (embedding)
         |
         v
    [2. RETRIEVE] --> Search vector database for similar documents
         |
         v
    [3. AUGMENT] ---> Add retrieved documents to the prompt
         |
         v
    [4. GENERATE] --> LLM generates answer using original question + retrieved context
         |
         v
    Answer (with source references)
    """, language=None)

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

    st.markdown("""
    ### How Fine-Tuning Works
    """)

    st.code("""
    Domain-Specific Dataset (e.g., 8,000+ financial Q&A pairs)
         |
         v
    [1. PREPARE] -----> Format data as instruction/response pairs
         |
         v
    [2. TRAIN] -------> Update model weights on your data
         |                 (Full fine-tuning or parameter-efficient: LoRA/QLoRA)
         v
    [3. EVALUATE] ----> Test on held-out data, measure accuracy
         |
         v
    [4. DEPLOY] ------> Use the specialized model for inference
         |
         v
    Model with NEW capabilities (reasoning, calculations, domain expertise)
    """, language=None)

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


def slide_head_to_head():
    """Slide 11: Head-to-Head Comparison"""
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


def slide_decision_framework():
    """Slide 13: Decision Framework"""
    st.markdown('<p class="slide-title">Decision Framework: When to Use What</p>', unsafe_allow_html=True)

    st.code("""
                          Does the task require
                        NEW REASONING SKILLS?
                       /                      \\
                     YES                       NO
                      |                         |
            Does it need               Does it need
           FRESH/DYNAMIC data?        FRESH/DYNAMIC data?
            /           \\              /           \\
          YES            NO          YES            NO
           |              |           |              |
        HYBRID      FINE-TUNE        RAG       PROMPT ENG.
     (FT + RAG)    (Best accuracy) (Dynamic)   (Quick start)
    """, language=None)

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
    st.markdown("### Monitoring & Experiment Tracking")
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.markdown("""
        <div class="tool-card">
        <strong>Weights & Biases (W&B)</strong><br/>
        Track experiments, compare runs, visualize training metrics.
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown("""
        <div class="tool-card">
        <strong>MLflow</strong><br/>
        Open-source ML lifecycle management. Model registry, versioning.
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown("""
        <div class="tool-card">
        <strong>TensorBoard</strong><br/>
        Built-in with PyTorch/TensorFlow. Real-time training visualization.
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

        ### Customer Service
        - **Domain-Specific Chatbots:** Trained on your product knowledge
        - **Technical Support:** Understands product-specific terminology
        - **Multilingual Support:** Fine-tuned for specific language pairs
        """)

    st.info("""
    **Common pattern:** Start with RAG for quick wins, then add fine-tuning
    where accuracy gaps appear. The data from RAG usage often becomes the
    training data for fine-tuning.
    """)


def slide_hybrid():
    """Slide 18: The Hybrid Approach"""
    st.markdown('<p class="slide-title">The Hybrid Approach: Best of Both Worlds</p>', unsafe_allow_html=True)

    st.code("""
    User Question + Financial Table
         |
         +---------+-----------+
         |                     |
         v                     v
    [RETRIEVE]            [FINE-TUNED
     from Vector DB]       MODEL PROCESSES
         |                 TABLE DATA]
         v                     |
    Retrieved Context          |
         |                     |
         +--------+   +-------+
                  |   |
                  v   v
            [FINE-TUNED MODEL]
            [+ Retrieved Context]
                    |
                    v
            Answer with:
            - Domain reasoning (from fine-tuning)
            - Fresh context (from RAG)
            - Source citations
    """, language=None)

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
        "Fine-tuning is essential when you need specialized reasoning, calculations, or consistent behavior",
        "RAG is valuable for dynamic knowledge, source citations, and quick deployment",
        "The best production systems often combine both approaches (hybrid)",
        "Modern tools (Unsloth, LoRA, QLoRA) make fine-tuning accessible without massive infrastructure",
        "Start with RAG for quick wins, add fine-tuning where accuracy gaps appear",
        "The cost of fine-tuning is an investment - higher accuracy means fewer costly errors",
    ]

    for i, takeaway in enumerate(takeaways, 1):
        st.markdown(f"""
        <div class="green-box">
        <strong>{i}.</strong> {takeaway}
        </div>
        """, unsafe_allow_html=True)


def slide_demo_intro():
    """Slide 21: Live Demo Introduction"""
    st.markdown('<p class="slide-title">Live Demo: See the Difference</p>', unsafe_allow_html=True)

    st.markdown("""
    ### What we'll demonstrate:
    """)

    col1, col2, col3 = st.columns(3)

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

    with col3:
        st.markdown("""
        <div class="tool-card">
        <h3>Demo 3: Benchmarks at Scale</h3>
        <p>Pre-computed results across <strong>8,281 test cases</strong>.</p>
        <p>See the aggregate numbers that prove fine-tuning's advantage.</p>
        <p><em>Navigate to: Benchmark Results page</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.success("**Use the sidebar navigation** to switch to the demo pages. Let's see fine-tuning in action!")


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
    ("Head-to-Head Comparison", slide_head_to_head),
    ("When RAG Falls Short", slide_rag_falls_short),
    ("Decision Framework", slide_decision_framework),
    ("Fine-Tuning Tools", slide_finetuning_tools),
    ("RAG Tools", slide_rag_tools),
    ("Data & Evaluation Tools", slide_data_preparation_tools),
    ("Real-World Use Cases", slide_use_cases),
    ("The Hybrid Approach", slide_hybrid),
    ("Cost & ROI", slide_cost_roi),
    ("Key Takeaways", slide_key_takeaways),
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
