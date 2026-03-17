# State-of-the-Art 2026 Upgrade Plan: Fine-Tuning vs RAG

## Context

The project is a strong interactive demo comparing FT vs RAG vs Hybrid across 4 tasks (sentiment, numerical reasoning, financial ratios, spam detection). To reach **2026 SOTA**, it needs to address gaps identified by 10 academic papers in the `papers/` folder: missing approaches (prompt engineering, distillation, agentic RAG), weak evaluation methodology (no LLM-as-judge, no statistical rigor, no adversarial testing), outdated models (Llama2), tiny test sets, and no efficiency frontier analysis.

### Papers Reference

| # | Paper | Venue | Key Insight |
|---|-------|-------|-------------|
| 1 | Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs | EMNLP 2024 | RAG consistently outperforms unsupervised FT; LLMs struggle to learn new facts through FT |
| 2 | Should We Fine-Tune or RAG? Evaluating Techniques for Dialogue | INLG 2024 | No universal best technique; human evaluation essential (auto metrics mislead) |
| 3 | Fine Tuning LLMs for Enterprise: Practical Guidelines | HCLTech 2024 | QLoRA rank selection, data prep recipes, FT+RAG pipeline evaluation |
| 4 | DSL Code Generation: Fine-Tuning vs Optimized RAG | Microsoft 2024 | Optimized RAG can match FT quality; RAG better for unseen APIs |
| 5 | Finetune-RAG: Resist Hallucination in RAG | Pints AI 2025 | 21.2% accuracy improvement; Bench-RAG LLM-as-judge pipeline |
| 6 | Fine-Tuning with RAG for Improving LLM Learning of New Skills | ICLR 2026 | RAG-to-FT distillation; 10-60% fewer tokens at inference |
| 7 | Domain-Driven LLM Development: RAG and Fine-Tuning Practices | KDD 2024 | Cost/ROI analysis methodology; combined architecture benchmarking |
| 8 | FT vs RAG for Less Popular Knowledge | SIGIR-AP 2024 (51 citations) | RAG surpasses FT for long-tail knowledge; proposes Stimulus RAG |
| 9 | RAG vs Fine-Tuning vs Prompt Engineering | IBM/IJCTEC 2025 | Three-way comparison including prompt engineering |
| 10 | Fine-tuning LLM using RLHF for Therapy Chatbot | KTH 2023 | RLHF for domain specialization; ethics discussion |

---

## Phase 1: Foundation Fixes & Quick Wins (1-2 weeks)

### 1.1 Fix FinQA-7B & Upgrade to Modern Models
**Papers:** #3 (HCLTech QLoRA), #8 (SIGIR-AP model scale)

- Replace Llama2-7B with **Qwen2.5-7B** or **Llama3.1-8B** (native Ollama support, no LoRA import crashes)
- Use a pre-quantized GGUF financial model instead of fragile LoRA adapter import
- Files: `docker-entrypoint.sh`, `docker-compose.yml`, `app/demo_utils.py` (LLM_MODEL/FINETUNED_LLM_MODEL defaults), `app/benchmark.py` (labels)

### 1.2 Add Prompt Engineering as 5th Approach
**Papers:** #9 (IBM: three-way comparison), #2 (INLG: task-dependent)

Every 2026 benchmark includes prompt engineering (few-shot + CoT) as the "zero-cost" baseline.

- Add `call_prompted_model()` in `app/demo_utils.py` -- base model + 3-4 in-context examples + chain-of-thought
- Add `run_prompted_sentiment()` for classification tasks (zero-shot NLI via `facebook/bart-large-mnli`)
- Add `"prompted"` to all model lists in `app/benchmark.py`
- Update colors/labels in `app/pages/9_Benchmark_Results.py`

### 1.3 Expand Test Cases (53 -> 170+)
**Papers:** #1 (EMNLP: knowledge score), #8 (SIGIR-AP: popularity tiers)

- Sentiment: 20 -> **60** (add domain inversions, rare financial language, adversarial)
- Spam: 20 -> **40** (add sophisticated phishing, borderline cases)
- Numerical: 5 -> **15** (add multi-hop, unit conversion)
- Financial ratios: 8 -> **15** (add complex multi-step)
- Add `"difficulty": "easy|medium|hard"` to every case
- File: `data/benchmark_test_cases.json`

### 1.4 Statistical Rigor
**Papers:** #1 (EMNLP), #8 (SIGIR-AP: 51-citation methodology)

- Add bootstrap 95% confidence intervals in `src/evaluation/metrics.py`
- Add McNemar's paired significance test (model A vs model B)
- Display CI error bars + significance stars in `app/pages/9_Benchmark_Results.py`

### 1.5 Pareto Efficiency Frontiers
**Papers:** #7 (KDD: cost/ROI), #6 (ICLR 2026: efficiency analysis)

- Add Plotly scatter: accuracy vs cost (bubble size = latency) per approach
- Draw Pareto frontier line connecting non-dominated points
- Add accuracy vs latency chart (bubble size = cost)
- File: `app/pages/9_Benchmark_Results.py` (new `render_pareto_frontier()`)

---

## Phase 2: Core SOTA Features (2-4 weeks)

### 2.1 LLM-as-Judge Evaluation
**Papers:** #5 (Pints AI: Bench-RAG with GPT-4o judge), #2 (INLG: auto metrics mislead)

**The single most impactful upgrade for 2026 credibility.**

- New file `app/llm_judge.py`: structured scoring (correctness, reasoning quality, faithfulness, each 1-5)
- Use a larger Ollama model as judge (Qwen2.5-32B or Llama3.1-70B, fallback to 7B)
- Optional `--with-judge` mode in benchmark runner
- Display judge assessments in a new tab per section in results page

### 2.2 Hallucination & Adversarial Stress Testing
**Papers:** #5 (Pints AI: 21.2% improvement with hallucination resistance), #1 (EMNLP: FT struggles with new facts)

- Add 30 adversarial test cases: 10 noisy retrieval, 10 knowledge conflict, 10 out-of-distribution
- Add `retrieve_with_noise(query, noise_level)` in `app/rag_engine.py` -- replaces fraction of results with irrelevant docs
- Add `run_adversarial_benchmark()` in `app/benchmark.py` -- tests all models under clean/33%/66% noise
- New page `app/pages/11_Robustness_Testing.py` -- robustness degradation curves per approach

### 2.3 RAG Distillation Experiment (ICLR 2026)
**Papers:** #6 (ICLR 2026: "retrieval benefits can be internalized through targeted fine-tuning")

**The most novel technique -- directly from a top-tier 2026 venue.**

- New file `app/rag_distillation.py`:
  1. Run RAG on N training questions, collect (question, answer) pairs
  2. Fine-tune base model on these pairs WITHOUT retrieval context
  3. At inference: RAG-quality answers at FT speed (no retrieval overhead)
- Add `"distilled"` as 6th approach in benchmark
- Display token savings: distilled tokens/query vs RAG tokens/query

### 2.4 Stimulus RAG
**Papers:** #8 (SIGIR-AP: "surpasses FT without costly data augmentation", 51 citations)

- Add `retrieve_with_stimulus()` in `app/rag_engine.py`: two-hop retrieval (retrieve -> extract entities -> re-retrieve)
- Add `call_stimulus_rag_model()` in `app/demo_utils.py`
- Add `"stimulus_rag"` as 7th approach -- tests whether enhanced retrieval can match FT

### 2.5 Upgrade RAG Pipeline
**Papers:** #4 (Microsoft: optimized RAG), #8 (SIGIR-AP: retrieval quality critical)

- Replace `all-MiniLM-L6-v2` with `BAAI/bge-small-en-v1.5` (better financial retrieval)
- Add hybrid search: vector similarity + BM25 keyword matching
- Add cross-encoder reranking (top-10 -> rerank -> top-3)
- Reduce chunks to 200 words / 75 overlap for denser retrieval
- Expand docs from 12 files to **30+ files** (add SEC 10-K, earnings transcripts, analyst reports)

### 2.6 Interactive Decision Framework Page
**Papers:** #7 (KDD: Lab 3), #9 (IBM: practical guide), #3 (HCLTech: recipes)

- New page `app/pages/12_Decision_Framework.py`
- User inputs: task type, data availability, latency budget, cost budget, knowledge freshness
- Output: recommended approach with benchmark evidence + paper citations
- Interactive Plotly Sankey/treemap diagram

---

## Phase 3: Differentiating Features (4+ weeks)

### 3.1 Agentic RAG
**Papers:** #6 (ICLR 2026: ReAct/StateAct agents)

- New file `app/agentic_rag.py`: ReAct-style agent (THINK -> ACT -> OBSERVE -> re-retrieve if needed)
- Add as 8th approach -- higher latency/tokens but better accuracy on hard questions
- Interesting Pareto frontier point

### 3.2 Explainability & Attribution
**Papers:** #2 (INLG: integrated gradients for attribution)

- New file `app/explainability.py`: attention-based token attribution for BERT/FinBERT
- New page `app/pages/13_Explainability.py`: side-by-side heatmaps for FT vs RAG
- Show retrieval chain (query -> docs -> answer) with relevance scores

### 3.3 Knowledge Freshness & Temporal Testing
**Papers:** #1 (EMNLP: FT struggles with new facts), #8 (SIGIR-AP: popular vs unpopular)

- Split test cases by temporal distribution: in-training vs post-training vs evolving
- Add 2024-2026 dated documents to RAG corpus that FT model couldn't have seen
- Expected: RAG dominates post-training, FT dominates in-distribution patterns

### 3.4 Multi-Scale Model Comparison
**Papers:** #8 (SIGIR-AP: 80M to 11B), #6 (ICLR 2026: 7B/14B)

- Parameterize model sizes: small (3.8B), medium (7B), large (70B)
- Chart how FT vs RAG gap changes with scale
- Find the cross-over point

### 3.5 Human Evaluation Interface
**Papers:** #2 (INLG: human eval essential)

- New page `app/pages/14_Human_Eval.py`: blinded side-by-side outputs, user ratings (1-5)
- Store in `data/human_eval_results.json`
- Compare human vs automated metric rankings

---

## Updated Approach Taxonomy (8 approaches)

| # | Approach | Training? | Retrieval at Inference? | Phase |
|---|---------|-----------|------------------------|-------|
| 1 | Base | No | No | Existing |
| 2 | Prompt Engineering | No | No (examples in prompt) | Phase 1 |
| 3 | Fine-Tuned | Yes (LoRA/QLoRA) | No | Existing (fix) |
| 4 | RAG | No | Yes | Existing (upgrade) |
| 5 | Stimulus RAG | No | Yes (two-hop) | Phase 2 |
| 6 | Hybrid (FT+RAG) | Yes | Yes | Existing |
| 7 | Distilled | Yes (from RAG teacher) | No | Phase 2 |
| 8 | Agentic RAG | No | Yes (multi-step) | Phase 3 |

## Updated Metric Taxonomy

| Category | Metrics | Phase |
|----------|---------|-------|
| **Correctness** | Accuracy, F1, MAPE, 95% CI, McNemar significance | Phase 1 |
| **Correctness** | LLM Judge (correctness, reasoning, faithfulness 1-5) | Phase 2 |
| **Efficiency** | Latency, throughput, cost/1K queries, token efficiency | Existing |
| **Robustness** | Accuracy under noisy retrieval (0%/33%/66%), OOD accuracy | Phase 2 |
| **Trade-off** | Pareto frontier (accuracy vs cost vs latency) | Phase 1 |
| **Explainability** | Token attribution, retrieval chain visualization | Phase 3 |
| **Human** | Human eval scores, human-vs-auto agreement | Phase 3 |

## Presentation Updates (app/pages/0_Presentation.py)

Add slides:
- "The 2026 Landscape" -- cite all 10 papers
- "Beyond Binary: 8 Approaches on the Pareto Frontier"
- "When RAG Breaks" -- adversarial results (Paper #5)
- "The Distillation Surprise" -- RAG quality at FT speed (Paper #6)
- "What Auto-Metrics Miss" -- human vs automated (Paper #2)

## Key Files Changed

| File | Phase | Summary |
|------|-------|---------|
| `docker-entrypoint.sh` | 1 | Fix model import, upgrade to modern models |
| `docker-compose.yml` | 1 | New model env vars |
| `app/demo_utils.py` | 1-2 | Prompt eng, stimulus RAG, distilled model functions |
| `app/benchmark.py` | 1-2 | New approaches, adversarial tests, CI, expanded cases |
| `app/rag_engine.py` | 1-2 | Upgrade embeddings, hybrid search, reranking, noise injection |
| `app/pages/9_Benchmark_Results.py` | 1-2 | Pareto charts, CI bars, significance, judge tab |
| `src/evaluation/metrics.py` | 1 | Bootstrap CI, McNemar, judge agreement |
| `data/benchmark_test_cases.json` | 1-2 | 53 -> 170+ cases with difficulty + adversarial |
| `data/documents/` | 2 | 12 -> 30+ financial documents |
| `app/llm_judge.py` | 2 | NEW: LLM-as-judge evaluation |
| `app/rag_distillation.py` | 2 | NEW: RAG-to-FT distillation |
| `app/pages/11_Robustness_Testing.py` | 2 | NEW: Adversarial visualization |
| `app/pages/12_Decision_Framework.py` | 2 | NEW: Interactive decision tree |
| `app/agentic_rag.py` | 3 | NEW: ReAct-style agent |
| `app/explainability.py` | 3 | NEW: Attribution analysis |
| `app/pages/13_Explainability.py` | 3 | NEW: Heatmaps |
| `app/pages/14_Human_Eval.py` | 3 | NEW: Blinded human rating |
| `app/pages/0_Presentation.py` | 2-3 | New slides for 2026 findings |

## Verification

- **Phase 1:** Run `docker compose up --build`, verify all 5 approaches execute on sentiment benchmark, confirm CI error bars display, confirm Pareto chart renders
- **Phase 2:** Run adversarial benchmark, verify degradation curves, run LLM judge on 10 cases, verify distilled model produces answers without retrieval
- **Phase 3:** Test agentic RAG on 5 hard questions, verify attribution heatmaps render, test human eval UI stores ratings
