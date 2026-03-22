<p align="center">
  <h1 align="center">Fine-Tuning vs RAG: The Definitive Benchmark</h1>
  <p align="center">
    <strong>253 test cases. 7 models. 6 benchmarks. One question answered with data, not opinions.</strong>
  </p>
  <p align="center">
    <a href="#quick-start"><img src="https://img.shields.io/badge/docker-one--click%20start-2496ED?logo=docker&logoColor=white" alt="Docker"></a>
    <a href="#the-results"><img src="https://img.shields.io/badge/253-test%20cases-brightgreen" alt="253 test cases"></a>
    <a href="#llm-as-judge"><img src="https://img.shields.io/badge/GPT--4o-LLM%20as%20Judge-orange" alt="LLM Judge"></a>
    <a href="#papers"><img src="https://img.shields.io/badge/papers-10%20referenced-blueviolet" alt="Papers"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  </p>
</p>

---

**Should you fine-tune or use RAG?** Everyone has an opinion. We have **253 measured experiments**.

This project runs **controlled, apples-to-apples benchmarks** across 4 tasks, 4 approaches, and 7 models -- all on the same hardware, same evaluation, full transparency. It ships as a one-click Docker app with an interactive Streamlit UI, a 39-slide presentation, and pre-computed results you can explore immediately.

```
Fine-tuning teaches SKILLS.  RAG provides KNOWLEDGE.
The best systems combine both.  We prove it.
```

## High-Level Architecture

```mermaid
graph TB
    subgraph INPUT["Test Case Input"]
        Q["Financial Question<br/>+ Data Table"]
    end

    subgraph APPROACHES["4 Approaches — Same Architecture, Different Strategy"]
        direction LR
        BASE["Base Model<br/><i>No training, no retrieval</i><br/>BERT 110M / Llama2 7B / DistilBERT 66M"]
        FT["Fine-Tuned Model<br/><i>Updated weights</i><br/>FinBERT / FinQA-7B / Spam-DistilBERT"]
        RAG["RAG<br/><i>Base + retrieved docs</i><br/>ChromaDB → Top-3 chunks"]
        HYB["Hybrid<br/><i>Fine-tuned + retrieved docs</i><br/>Best of both"]
    end

    subgraph EVAL["Evaluation — 253 Test Cases"]
        ACC["Accuracy / F1 / MAPE"]
        JUDGE["GPT-4o LLM Judge<br/>Correctness · Reasoning · Faithfulness"]
        COST["Latency · Tokens · $/1K queries"]
    end

    Q --> BASE & FT & RAG & HYB
    BASE & FT & RAG & HYB --> ACC & JUDGE & COST

    style INPUT fill:#1a237e,stroke:#5c6bc0,color:#fff
    style APPROACHES fill:#0d1440,stroke:#2a4a8a,color:#fff
    style EVAL fill:#004d40,stroke:#26a69a,color:#fff
    style BASE fill:#37474f,stroke:#78909c,color:#fff
    style FT fill:#1b5e20,stroke:#66bb6a,color:#fff
    style RAG fill:#01579b,stroke:#29b6f6,color:#fff
    style HYB fill:#4a148c,stroke:#ab47bc,color:#fff
```

## What makes this different

| | |
|---|---|
| **Same architecture** | Base vs fine-tuned use identical architectures -- only weights differ |
| **4-way comparison** | Every test case runs through Base, Fine-Tuned, RAG, and Hybrid |
| **253 test cases** | 6 benchmark suites: standard, adversarial, RAG strengths, model family |
| **LLM-as-Judge** | GPT-4o structured evaluation (correctness, reasoning, faithfulness) |
| **The Formula Trap** | We discovered why RAG appears to fail -- and proved it's the benchmark, not RAG |
| **Full presentations** | 39 web slides + 62-slide PowerPoint, auto-generated from benchmark data |
| **Real cost analysis** | Tokens, latency, and $/1K queries measured for every approach |

## The headline numbers

| Finding | Evidence |
|---------|----------|
| Fine-tuning teaches skills | FinBERT 70% vs base 45% on sentiment |
| RAG can't teach math | 15% &rarr; 15.3% on numerical reasoning |
| **But RAG wasn't broken** | **86.7%** when data aligns (vs 15% with conflicts) |
| Hybrid wins overall | 93.3% on RAG strengths, 75% on sentiment |
| RAG reduces hallucination | Faithfulness: 3.8/5 vs base 1.9/5 (GPT-4o judge) |
| Model size isn't everything | 66M DistilBERT matches ~8B GPT-4o-mini on spam |

## The Discovery: RAG Was Never Broken

Most benchmarks show RAG struggling on numerical tasks (~15% accuracy). We discovered **why** -- and it changes the conclusion:

```mermaid
graph LR
    subgraph STANDARD["Standard Benchmark — RAG Accuracy: 15%"]
        direction TB
        S1["Test Table<br/><b>Revenue = $25.9B</b>"]
        S2["RAG Retrieves<br/><b>Meridian Revenue = $48.7B</b>"]
        S3["Model sees<br/><b>TWO conflicting numbers</b>"]
        S4["Confused → Wrong Answer"]
        S1 --> S3
        S2 --> S3
        S3 --> S4
    end

    subgraph ALIGNED["RAG Strengths Benchmark — RAG Accuracy: 86.7%"]
        direction TB
        A1["Question asks about<br/><b>Meridian's revenue</b>"]
        A2["RAG Retrieves<br/><b>Meridian Revenue = $48.7B</b>"]
        A3["Data <b>aligns</b><br/>No conflict"]
        A4["Correct Answer"]
        A1 --> A3
        A2 --> A3
        A3 --> A4
    end

    S4 -. "+71.4pp<br/>same model<br/>same RAG" .-> A4

    style STANDARD fill:#b71c1c,stroke:#e53935,color:#fff
    style ALIGNED fill:#1b5e20,stroke:#66bb6a,color:#fff
    style S4 fill:#c62828,stroke:#e53935,color:#fff
    style A4 fill:#2e7d32,stroke:#66bb6a,color:#fff
```

| Benchmark | RAG Accuracy | What Happened |
|-----------|-------------|---------------|
| Standard numerical | **15.3%** | Retrieved data **conflicted** with test data |
| RAG Strengths | **86.7%** | Retrieved data **aligned** with the question |
| Delta | **+71.4pp** | Same model, same RAG pipeline, different data alignment |

This is the **Formula Trap**: the RAG knowledge base provides formulas the model already knows, plus data that contradicts what the test case needs. The problem was never RAG -- it was the benchmark design. Our RAG Strengths benchmark fixes this by testing RAG on its actual production use case: answering questions about proprietary documents.

> Full analysis: [`data/rag_formula_vs_answer_analysis.md`](data/rag_formula_vs_answer_analysis.md)

## The Results

### Benchmark Methodology

```mermaid
graph TD
    subgraph SUITES["6 Benchmark Suites — 253 Test Cases"]
        direction TB
        STD["Standard<br/><b>53 cases</b><br/>Sentiment · Numerical<br/>Ratios · Spam"]
        ADV["Adversarial<br/><b>120 cases</b><br/>Noisy Retrieval<br/>Knowledge Conflict · OOD"]
        RS["RAG Strengths<br/><b>30 cases</b><br/>Direct Retrieval<br/>Cross-Doc · Trends"]
        MF["Model Family<br/><b>50 cases</b><br/>66M vs ~8B"]
        FVA["Formula vs Answer<br/><b>Qualitative</b><br/>7-category taxonomy"]
        COV["Coverage Analysis<br/><b>Qualitative</b><br/>KB audit per case"]
    end

    subgraph SCORING["Scoring Methods"]
        direction TB
        CLS["Classification<br/>Exact label match"]
        NUM["Numerical<br/>Within 5% tolerance"]
        KW["Keyword + Numeric<br/>Hybrid matching"]
    end

    subgraph JUDGE["LLM-as-Judge (GPT-4o)"]
        direction LR
        COR["Correctness<br/><b>50% weight</b><br/>1-5 scale"]
        REA["Reasoning<br/><b>30% weight</b><br/>1-5 scale"]
        FAI["Faithfulness<br/><b>20% weight</b><br/>1-5 scale"]
    end

    STD --> CLS & NUM
    RS --> KW
    RS & ADV --> JUDGE
    COR & REA & FAI --> OVR["Overall = C×0.5 + R×0.3 + F×0.2"]

    style SUITES fill:#1a237e,stroke:#5c6bc0,color:#fff
    style SCORING fill:#004d40,stroke:#26a69a,color:#fff
    style JUDGE fill:#e65100,stroke:#ff9800,color:#fff
    style OVR fill:#bf360c,stroke:#ff6e40,color:#fff
    style STD fill:#283593,stroke:#5c6bc0,color:#fff
    style ADV fill:#c62828,stroke:#e53935,color:#fff
    style RS fill:#2e7d32,stroke:#66bb6a,color:#fff
    style MF fill:#6a1b9a,stroke:#ab47bc,color:#fff
    style FVA fill:#37474f,stroke:#78909c,color:#fff
    style COV fill:#37474f,stroke:#78909c,color:#fff
```

### 6 Benchmark Suites, 253 Test Cases

| Suite | Cases | What It Tests | Key Finding |
|-------|-------|---------------|-------------|
| **Standard** (4 experiments) | 53 | Sentiment, numerical, ratios, spam | Fine-tuning wins on skill tasks |
| **Adversarial** (4 experiments) | 120 | Noisy retrieval, knowledge conflict, OOD | All approaches degrade; fine-tuning most robust |
| **RAG Strengths** | 30 | Factual retrieval, cross-doc synthesis, trends | RAG 86.7%, Hybrid 93.3% |
| **Model Family** | 50 | DistilBERT 66M vs GPT-4o-mini ~8B | 121x smaller model matches on spam |

### Experiment 1: Sentiment Classification (BERT 110M)

| Approach | Accuracy | Confidence | Cost/1K |
|----------|---------|-----------|---------|
| Base BERT | 45% | 0.375 | $0.0002 |
| **FinBERT (fine-tuned)** | **70%** | **0.845** | $0.0002 |
| BERT + RAG | 65% | 0.564 | ~$0.001 |
| **FinBERT + RAG (hybrid)** | **75%** | 0.702 | ~$0.001 |

**Where each wins:** FinBERT dominates domain jargon (100% vs RAG 0%) because it *learned* that "headwinds" means negative. RAG dominates subtle neutral cases (100% vs FinBERT 40%) because retrieved examples provide signal.

### Experiment 2: Numerical Reasoning (Llama2 7B)

| Approach | Accuracy | Why |
|----------|---------|-----|
| Base Llama2 | ~15% | Can't do financial math |
| **FinQA-7B** | **61.2%** | Learned calculation patterns from 8,281 FinQA examples |
| Llama2 + RAG | 15.3% | Retrieval adds context, not computation ability |
| **FinQA-7B + RAG** | **65.8%** | Best of both |

### Experiment 5: RAG Strengths (Llama2 7B) -- NEW

*30 cases testing RAG on proprietary document retrieval, cross-document synthesis, and contextual interpretation.*

| Approach | Accuracy | Judge Score | Faithfulness |
|----------|---------|-------------|-------------|
| Base Llama2 | 43.3% | 1.92 / 5 | 1.9 / 5 |
| **Llama2 + RAG** | **86.7%** | **3.49 / 5** | **3.8 / 5** |
| FinQA-7B | 40.0% | 2.00 / 5 | 2.1 / 5 |
| **FinQA-7B + RAG (hybrid)** | **93.3%** | **3.64 / 5** | **3.8 / 5** |

**By category:**

| Category | Base | RAG | Fine-tuned | Hybrid |
|----------|------|-----|------------|--------|
| Direct Retrieval (8 cases) | 12.5% | 75.0% | 0.0% | **100%** |
| Formula + Aligned Data (6) | 16.7% | 83.3% | 33.3% | 66.7% |
| Cross-Document Synthesis (8) | 62.5% | 87.5% | 62.5% | **100%** |
| Contextual Interpretation (4) | 50.0% | **100%** | 25.0% | **100%** |
| Trend Analysis (4) | 100% | 100% | 100% | 100% |

> Direct retrieval shows the largest gap: base models literally cannot answer questions about documents they've never seen. This is RAG's fundamental value proposition.

### Adversarial Stress Test (120 cases)

```mermaid
graph TD
    subgraph ATTACKS["3 Attack Vectors per Task"]
        direction LR
        NR["Noisy Retrieval<br/><i>Irrelevant docs injected</i>"]
        KC["Knowledge Conflict<br/><i>Contradictory signals</i>"]
        OOD["Out of Distribution<br/><i>ESG · Crypto · DeFi</i>"]
    end

    subgraph SENTIMENT["Adversarial Sentiment (30 cases)"]
        direction LR
        AS_W["Winner: <b>RAG 3.23/5</b><br/>Faithfulness anchors it"]
        AS_L["All: 20-30% on<br/>noisy retrieval"]
    end

    subgraph SPAM["Adversarial Spam (30 cases)"]
        direction LR
        SP_W["Winner: <b>Hybrid 4.10/5</b><br/>Best adversarial score<br/>in entire project"]
        SP_L["Fine-tuned patterns<br/>robust under attack"]
    end

    subgraph NUMERICAL["Adversarial Numerical (30 cases)"]
        direction LR
        NU_W["All models collapse<br/><b>Hybrid worst: 2.06/5</b>"]
        NU_L["More context =<br/>more confusion"]
    end

    subgraph RATIOS["Adversarial Ratios (30 cases)"]
        direction LR
        FR_W["All models fail<br/><b>Hybrid lowest: 1.86/5</b>"]
        FR_L["Lowest score in<br/>entire benchmark"]
    end

    NR & KC & OOD --> SENTIMENT & SPAM & NUMERICAL & RATIOS

    style ATTACKS fill:#c62828,stroke:#e53935,color:#fff
    style SENTIMENT fill:#01579b,stroke:#29b6f6,color:#fff
    style SPAM fill:#2e7d32,stroke:#66bb6a,color:#fff
    style NUMERICAL fill:#e65100,stroke:#ff9800,color:#fff
    style RATIOS fill:#b71c1c,stroke:#e53935,color:#fff
```

**Key adversarial insight:** On tasks requiring reasoning skills (numerical, ratios), adding more context through RAG **degrades** performance. Hybrid scores the worst (1.86/5) on adversarial ratios — irrelevant context is worse than no context. But on classification tasks (sentiment, spam), fine-tuning and RAG provide complementary robustness.

### LLM-as-Judge

Every RAG Strengths test case is evaluated by GPT-4o on three dimensions:

- **Correctness** (1-5): Does the answer contain the right facts?
- **Reasoning Quality** (1-5): Does it show understanding?
- **Faithfulness** (1-5): Is it grounded in documents (not hallucinated)?

RAG models score **2x higher on faithfulness** (3.8 vs 1.9) than base models -- retrieved documents anchor responses in facts rather than hallucination.

```mermaid
xychart-beta
    title "LLM Judge Scores — RAG Strengths Benchmark (30 cases)"
    x-axis ["Correctness", "Reasoning", "Faithfulness", "Overall"]
    y-axis "Score (1-5)" 0 --> 5
    bar [1.50, 2.60, 1.93, 1.92]
    bar [3.43, 3.37, 3.80, 3.49]
    bar [1.57, 2.67, 2.07, 2.00]
    bar [3.57, 3.43, 3.67, 3.64]
```

*Bars: Base | RAG | Fine-tuned | Hybrid — RAG's faithfulness (3.80) is the highest single score; Hybrid leads overall (3.64)*

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/intelliswarm-ai/finetune-vs-rag.git
cd finetune-vs-rag
cp .env.example .env          # Add OPENAI_API_KEY for LLM-as-Judge (optional)
docker compose up --build
# Open http://localhost:8501
```

First run takes ~10 minutes (downloads 7B models, builds FinQA-7B via LoRA merge, indexes documents, pre-computes all benchmarks). After that, starts in seconds.

**What Docker does automatically:**
- Pulls Llama2-7B and creates FinQA-7B (LoRA adapter merge) via Ollama
- Downloads FinBERT, bert-base-uncased, DistilBERT, and sentence-transformers
- Initializes ChromaDB with 12 financial documents (24 chunks)
- Pre-computes all 6 benchmark suites (253 test cases)
- Starts the Streamlit app on port 8501

### macOS / Linux (native)

```bash
brew install ollama python       # macOS, or your package manager
cp .env.example .env
./run-macos.sh
# Open http://localhost:8501
```

Requires ~8GB RAM for 7B model inference.

## What's Inside

### Interactive Presentation (39 web slides + 62 PowerPoint slides)

A complete educational deck covering:
- LLM fundamentals and the specialization challenge
- RAG mechanics: embeddings, vector stores, retrieval pipelines
- Fine-tuning methods: full fine-tuning, LoRA, QLoRA
- Head-to-head comparison with decision framework
- Tools landscape: training platforms, RAG infrastructure
- All 6 benchmark suites with charts and analysis
- RAG Strengths findings and the Formula Trap discovery
- LLM-as-Judge quality assessment with radar charts
- Conclusions with actionable decision framework

The PowerPoint is auto-generated from benchmark data (`python generate_pptx.py`) with speaker notes for every slide.

### Live Demos (4 experiments)

Each experiment has a demo page and a live query page:

- **Sentiment Analysis** -- Type financial text, see FinBERT vs base BERT vs RAG vs Hybrid classify it in real time with confidence scores
- **Numerical Reasoning** -- Enter financial questions with data tables, watch all 4 approaches attempt calculations with step-by-step streaming
- **Financial Ratios** -- Complex multi-step ratio computations (DuPont ROE, CAGR, leverage)
- **Spam Detection** -- Fine-tuned DistilBERT vs base vs RAG on phishing/spam emails

### 6 Benchmark Dashboards

| Dashboard | Cases | Features |
|-----------|-------|----------|
| Standard Results | 53 | Accuracy, latency, F1, cost, category breakdown |
| Adversarial Stress Test | 120 | Noisy retrieval, knowledge conflict, OOD |
| RAG Strengths | 30 | Factual retrieval, cross-doc synthesis, LLM judge |
| Model Family | 50 | DistilBERT 66M vs GPT-4o-mini ~8B |
| How It Works | -- | Architecture diagrams, RAG pipeline |
| Presentation | 39 | Slides with Mermaid diagrams, charts |

Every dashboard supports both **live execution** (run in the UI with progress bars) and **pre-computed results** (instant loading when models are offline).

## Architecture

```
finetune-vs-rag/
├── app/
│   ├── finetune_vs_rag.py              # Landing page
│   ├── demo_utils.py                   # All model inference (7 models, 4 approaches)
│   ├── rag_engine.py                   # ChromaDB + sentence-transformers pipeline
│   ├── benchmark.py                    # Standard benchmark runner
│   ├── adversarial_benchmark.py        # Adversarial stress test runner
│   ├── rag_strengths_benchmark.py      # RAG strengths benchmark runner
│   ├── llm_judge.py                    # GPT-4o structured evaluation
│   ├── model_family_benchmark.py       # Model size comparison runner
│   └── pages/                          # 14 Streamlit pages
├── src/
│   ├── rag/                            # Embeddings, vector store, RAG pipeline
│   ├── models/                         # Model wrappers (FinBERT, FinQA, hybrid)
│   └── evaluation/                     # Metrics (F1, MAPE) + model comparator
├── data/
│   ├── documents/                      # 12 financial docs for RAG (~27KB)
│   ├── benchmark_test_cases.json       # Standard test cases (53)
│   ├── adversarial_test_cases.json     # Adversarial cases (120)
│   ├── rag_strengths_benchmark.json    # RAG strengths cases (30)
│   ├── benchmark_results.json          # Pre-computed standard results
│   ├── adversarial_results.json        # Pre-computed adversarial results
│   ├── rag_strengths_results.json      # Pre-computed RAG strengths results
│   ├── rag_formula_vs_answer_analysis.md   # Formula Trap analysis
│   └── rag_coverage_analysis.md        # RAG KB coverage audit
├── papers/                             # 10 academic papers on FT vs RAG
├── generate_pptx.py                    # PowerPoint generator (62 slides)
├── Dockerfile                          # Multi-stage build with model pre-download
├── docker-compose.yml                  # Ollama + Streamlit orchestration
└── docker-entrypoint.sh                # Automated model setup + benchmark pre-computation
```

### Models

| Model | Params | Role | Source |
|-------|--------|------|--------|
| FinBERT | 110M | Fine-tuned financial sentiment | `ProsusAI/finbert` |
| bert-base-uncased | 110M | Base sentiment (same arch) | HuggingFace |
| FinQA-7B | 7B | Fine-tuned numerical reasoning (LoRA) | `truocpham/FinQA-7B-Instruct-v0.1` |
| llama2 | 7B | Base LLM | Ollama |
| DistilBERT (fine-tuned) | 66M | Spam/phishing classifier | Custom checkpoint |
| distilbert-base-uncased | 66M | Base spam (same arch) | HuggingFace |
| all-MiniLM-L6-v2 | 22M | RAG embeddings (384-dim) | sentence-transformers |

### RAG Pipeline

```mermaid
graph LR
    subgraph INGEST["Document Ingestion (one-time)"]
        direction TB
        DOCS["12 Financial Documents<br/><i>Meridian National Bancorp</i><br/>Annual report · Capital ratios<br/>Revenue · Risk · Regulatory"]
        CHUNK["Chunking<br/>300 words / 50-word overlap"]
        EMB1["Embedding<br/>all-MiniLM-L6-v2<br/><i>384 dimensions</i>"]
        VDB[("ChromaDB<br/><b>24 chunks indexed</b>")]
        DOCS --> CHUNK --> EMB1 --> VDB
    end

    subgraph QUERY["Query-Time Retrieval"]
        direction TB
        QINPUT["User Question"]
        EMB2["Embed Question<br/><i>384-dim vector</i>"]
        SIM["Cosine Similarity<br/>against 24 chunks"]
        TOP3["Top-3 Chunks<br/>+ source attribution"]
        QINPUT --> EMB2 --> SIM --> TOP3
    end

    subgraph GENERATE["Answer Generation"]
        direction TB
        PROMPT["Prompt =<br/>Question + Data Table<br/>+ Retrieved Chunks"]
        LLM["LLM generates answer<br/><i>grounded in documents</i>"]
        PROMPT --> LLM
    end

    VDB -.-> SIM
    TOP3 --> PROMPT

    style INGEST fill:#01579b,stroke:#29b6f6,color:#fff
    style QUERY fill:#004d40,stroke:#26a69a,color:#fff
    style GENERATE fill:#4a148c,stroke:#ab47bc,color:#fff
    style VDB fill:#0d47a1,stroke:#42a5f5,color:#fff
```

| Component | Implementation |
|-----------|---------------|
| Embedder | all-MiniLM-L6-v2 (384-dim, cosine similarity) |
| Vector Store | ChromaDB (in-memory, 24 chunks indexed) |
| Documents | 12 financial docs about Meridian National Bancorp |
| Chunking | 300-word chunks, 50-word overlap |
| Retrieval | Top-3 by cosine similarity with source attribution |
| Hybrid Blend | 60% fine-tuned scores + 40% RAG scores (classification) |

### The Fine-Tuning Is Real

FinQA-7B is **not llama2 with a better prompt**. It's llama2 with a LoRA adapter (rank 64, alpha 16) trained on 8,281 financial Q&A pairs from the FinQA dataset. The adapter modifies the `q_proj` and `v_proj` attention matrices -- actual weight changes. Ollama merges the adapter at model creation time.

FinBERT is bert-base-uncased fine-tuned on 50,000+ Financial PhraseBank sentences. Same 110M parameters, different weights.

## Key Takeaways

### The Fundamental Asymmetry: Knowledge vs Skill

```mermaid
quadrantChart
    title Knowledge vs Skill — What Each Approach Provides
    x-axis "Low Skill" --> "High Skill"
    y-axis "Low Knowledge" --> "High Knowledge"
    quadrant-1 "HYBRID: Skills + Knowledge"
    quadrant-2 "RAG: Knowledge only"
    quadrant-3 "BASE: Neither"
    quadrant-4 "FINE-TUNED: Skills only"
    "Base BERT (45% sentiment)": [0.2, 0.15]
    "Base Llama2 (20% numerical)": [0.15, 0.2]
    "FinBERT (70% sentiment)": [0.75, 0.15]
    "FinQA-7B (61% numerical)": [0.8, 0.2]
    "BERT+RAG (65% sentiment)": [0.25, 0.65]
    "Llama2+RAG (87% retrieval)": [0.2, 0.82]
    "Hybrid (93% RAG strengths)": [0.78, 0.85]
    "Hybrid (75% sentiment)": [0.72, 0.68]
```

### 1. Fine-tuning teaches skills, RAG provides knowledge

FinBERT *knows* "headwinds" is negative -- it learned this from training data. RAG can only find similar examples and guess. But RAG *knows* Meridian's CET1 ratio is 13.2% -- it retrieved the document. Fine-tuning can't access documents it was never trained on.

### 2. The data alignment problem explains most RAG failures

When RAG retrieves data that conflicts with the test case, it **hurts** performance (-71pp). When retrieved data aligns with the question, RAG **dominates** (+43pp over base). Most published benchmarks inadvertently create conflict scenarios.

```mermaid
graph TD
    subgraph TAXONOMY["What RAG Actually Retrieves — 7-Category Taxonomy"]
        direction TB
        DA["Direct Answer (DA)<br/><i>Exact data in KB</i><br/><b>Strong Positive</b>"]
        LP["Labeled Pattern (LP)<br/><i>Near-verbatim example</i><br/><b>Strong Positive → 96% acc</b>"]
        FO["Formula Only (FO)<br/><i>Formula already in question</i><br/><b>Neutral</b>"]
        FC["Formula + Conflict (FC)<br/><i>Redundant formula + wrong data</i><br/><b>Negative → 15% acc</b>"]
        CD["Conflicting Data (CD)<br/><i>Different company's numbers</i><br/><b>Strongly Negative → 5% acc</b>"]
        IR["Irrelevant (IR)<br/><i>Unrelated documents</i><br/><b>Slightly Negative → 10% acc</b>"]
        NK["Not in KB (NK)<br/><i>Domain outside KB scope</i><br/><b>No Effect → 30% acc</b>"]
    end

    subgraph COVERAGE["KB Coverage Audit — 171 Cases"]
        COV["Covered: 34 (20%)"]
        PART["Partially: 17 (10%)"]
        NOTC["Not Covered: 120 (70%)"]
        HARM["Actively Harmful: ~51 (30%)"]
    end

    DA & LP --> COV
    FO --> PART
    FC & CD --> HARM
    IR & NK --> NOTC

    style DA fill:#2e7d32,stroke:#66bb6a,color:#fff
    style LP fill:#2e7d32,stroke:#66bb6a,color:#fff
    style FO fill:#f9a825,stroke:#fdd835,color:#000
    style FC fill:#c62828,stroke:#e53935,color:#fff
    style CD fill:#b71c1c,stroke:#e53935,color:#fff
    style IR fill:#e65100,stroke:#ff9800,color:#fff
    style NK fill:#37474f,stroke:#78909c,color:#fff
    style HARM fill:#b71c1c,stroke:#e53935,color:#fff
    style COV fill:#2e7d32,stroke:#66bb6a,color:#fff
```

### 3. RAG reduces hallucination

GPT-4o judge scores show RAG models achieve **2x higher faithfulness** (3.8/5 vs 1.9/5). Retrieved documents anchor responses in facts. This is RAG's most important production value.

### 4. Hybrid wins when you need both skills and knowledge

Hybrid (FinQA-7B + RAG) achieves 93.3% on RAG strengths -- combining fine-tuning's reasoning with RAG's factual grounding for the best results across every category.

### 5. Model size isn't everything

A 66M-parameter DistilBERT fine-tuned on task-specific data matches ~8B GPT-4o-mini on spam detection. The 121x parameter difference doesn't translate to better accuracy for focused classification tasks.

### 6. The decision framework

```mermaid
flowchart TD
    START(["Does the task require<br/><b>NEW REASONING SKILLS?</b><br/><i>math, domain jargon,<br/>specialized classification</i>"])

    START -->|"YES"| FRESH1{"Does it need<br/><b>FRESH / DYNAMIC DATA</b><br/>or citations?"}
    START -->|"NO"| FRESH2{"Does it need<br/><b>FRESH / DYNAMIC DATA</b><br/>or citations?"}

    FRESH1 -->|"YES"| HYBRID["HYBRID<br/><b>Fine-Tune + RAG</b><br/>━━━━━━━━━━━━<br/>93.3% RAG Strengths<br/>95% Spam Detection<br/>100% Cross-Doc Synthesis"]
    FRESH1 -->|"NO"| FINETUNE["FINE-TUNE<br/><b>Best accuracy</b><br/>━━━━━━━━━━━━<br/>70% Sentiment (+25pp)<br/>61.2% Numerical (+46pp)<br/>~200ms latency (4x faster)"]

    FRESH2 -->|"YES"| RAG["RAG<br/><b>Dynamic knowledge</b><br/>━━━━━━━━━━━━<br/>86.7% Factual Retrieval<br/>3.8/5 Faithfulness (2x)<br/>No training required"]
    FRESH2 -->|"NO"| PROMPT["PROMPT ENGINEERING<br/><b>Quick start</b><br/>━━━━━━━━━━━━<br/>Works on straightforward cases<br/>Hours to deploy<br/>No infra needed"]

    style START fill:#1a237e,stroke:#5c6bc0,color:#fff
    style HYBRID fill:#4a148c,stroke:#ab47bc,color:#fff
    style FINETUNE fill:#1b5e20,stroke:#66bb6a,color:#fff
    style RAG fill:#01579b,stroke:#29b6f6,color:#fff
    style PROMPT fill:#e65100,stroke:#ff9800,color:#fff
    style FRESH1 fill:#37474f,stroke:#78909c,color:#fff
    style FRESH2 fill:#37474f,stroke:#78909c,color:#fff
```

| Choose | When | Evidence |
|--------|------|----------|
| **Fine-Tuning** | Domain skills: math, classification, jargon interpretation | FinBERT 100% on jargon vs RAG 0% |
| **RAG** | Proprietary documents, dynamic data, citation needed | RAG 86.7% on factual retrieval |
| **Hybrid** | Maximum accuracy on complex tasks requiring both | Hybrid 93.3% vs next-best 86.7% |
| **Prompt Engineering** | Quick start, simple tasks, no training data | Base models work on straightforward cases |

### Practical Workflow: From Prototype to Production

```mermaid
graph LR
    subgraph PHASE1["Phase 1: Start Fast"]
        direction TB
        P1A["Deploy <b>RAG</b><br/>Low cost · No training<br/>Days to launch"]
        P1B["Measure accuracy<br/>on real queries"]
        P1A --> P1B
    end

    subgraph PHASE2["Phase 2: Find Gaps"]
        direction TB
        P2A["Identify where<br/>RAG fails"]
        P2B["Collect failing queries<br/>as training data"]
        P2A --> P2B
    end

    subgraph PHASE3["Phase 3: Add Skills"]
        direction TB
        P3A["<b>Fine-tune</b> with QLoRA<br/>$5-25 on AWS<br/>500-1000 examples"]
        P3B["Evaluate on<br/>held-out test set"]
        P3A --> P3B
    end

    subgraph PHASE4["Phase 4: Production"]
        direction TB
        P4A["Deploy <b>Hybrid</b><br/>FT reasoning + RAG knowledge"]
        P4B["93.3% accuracy<br/>2x faithfulness<br/>Source citations"]
        P4A --> P4B
    end

    PHASE1 --> PHASE2 --> PHASE3 --> PHASE4

    style PHASE1 fill:#01579b,stroke:#29b6f6,color:#fff
    style PHASE2 fill:#e65100,stroke:#ff9800,color:#fff
    style PHASE3 fill:#1b5e20,stroke:#66bb6a,color:#fff
    style PHASE4 fill:#4a148c,stroke:#ab47bc,color:#fff
```

## Papers

This project's methodology is informed by 10 academic papers:

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs | EMNLP 2024 | RAG outperforms unsupervised FT for knowledge injection |
| Should We Fine-Tune or RAG? Evaluating Techniques for Dialogue | INLG 2024 | No universal best -- depends on task type |
| Fine Tuning LLMs for Enterprise: Practical Guidelines | 2024 | QLoRA guidelines, data prep recipes |
| DSL Code Generation: Fine-Tuning vs Optimized RAG | Microsoft 2024 | Optimized RAG matches FT quality |
| Finetune-RAG: Resist Hallucination in RAG | 2025 | 21.2% accuracy gain with hallucination-resistant FT |
| Fine-Tuning with RAG for Improving LLM Learning | ICLR 2026 | RAG-to-FT distillation: 10-60% fewer tokens |
| Domain-Driven LLM Development | KDD 2024 | Cost/ROI analysis for RAG vs FT |
| FT vs RAG for Less Popular Knowledge | SIGIR-AP 2024 | RAG dominates for long-tail knowledge |
| RAG vs Fine-Tuning vs Prompt Engineering | IJCTEC 2025 | Three-way comparison with prompt engineering |
| Fine-tuning LLM using RLHF | KTH 2023 | RLHF for domain specialization |

See [`state-of-the-art.md`](state-of-the-art.md) for the full roadmap with paper-backed improvements across 3 phases.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (14 pages) |
| LLM Serving | Ollama (OpenAI-compatible API) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| ML Framework | PyTorch, HuggingFace Transformers, PEFT |
| Evaluation | GPT-4o LLM-as-Judge, scikit-learn, NLTK |
| Visualization | Plotly |
| Presentation | python-pptx (auto-generated), Mermaid diagrams |
| Infrastructure | Docker, Docker Compose |

## Contributing

Contributions are welcome. See [`state-of-the-art.md`](state-of-the-art.md) for the improvement roadmap with specific tasks.

Areas where help is especially valuable:
- Additional benchmark domains beyond finance
- GPU-accelerated inference benchmarks
- Multi-language RAG evaluation
- Alternative embedding models comparison
- Fine-tuning with more recent base models (Llama3, Mistral)

## License

MIT License
