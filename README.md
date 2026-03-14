# Fine-Tuning vs RAG: Live Demo

A live, interactive demonstration comparing **Fine-Tuned LLMs** vs **RAG (Retrieval-Augmented Generation)** for financial services tasks.

Includes a full **22-slide presentation** and **live demos** showing why fine-tuning delivers superior results for specialized tasks.

## Quick Start (Docker)

```bash
# 1. (Optional) Add your API keys for live LLM comparison
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 2. Build and run
docker compose up --build

# 3. Open http://localhost:8501
```

That's it. FinBERT is pre-downloaded in the image - the demo starts instantly.

## Quick Start (Local)

```bash
pip install streamlit plotly pandas python-dotenv openai torch transformers
cp .env.example .env
streamlit run app/app.py
```

## What's Inside

### Presentation

Navigate with Previous/Next buttons in the Streamlit app.

| Slides | Content |
|--------|---------|
| 1-4 | LLMs, the specialization challenge |
| 5-7 | RAG: how it works, benefits, **limitations** |
| 8-10 | Fine-tuning: how it works, methods (LoRA/QLoRA), **key benefits** |
| 11-13 | Head-to-head comparison, decision framework |
| **14-16** | **Tools**: fine-tuning platforms, RAG infrastructure, data prep & eval |
| 17-20 | Use cases, hybrid approach, cost/ROI, key takeaways |
| 21-22 | Live demo introduction |

### Live Demos

**Sentiment Analysis (apples-to-apples comparison):**
- **FinBERT** (fine-tuned on 50K+ financial sentences) vs **RAG** (retrieve similar examples + classify)
- Same inputs, same task, same output format - different approach
- Shows speed (8ms vs 580ms), confidence, and accuracy differences

**Numerical Reasoning (3-way comparison):**
- Fine-Tuned (FinQA-7B) vs RAG vs Base LLM
- Financial calculations: revenue growth, D/E ratios, efficiency ratios
- Uses OpenAI API when available, realistic simulations otherwise

**Benchmarks:**
- Pre-computed results across 8,281 FinQA test cases
- Interactive charts and comparison tables

## Demo Modes

| Mode | Requirements | What's Live |
|------|-------------|-------------|
| **Full live** | Docker + `OPENAI_API_KEY` in `.env` | FinBERT + API calls |
| **FinBERT only** | Docker (no API key needed) | FinBERT sentiment |
| **Simulated** | Just `streamlit` | Realistic pre-computed responses |

## Project Structure

```
finetune-vs-rag/
├── Dockerfile                  # Containerized demo with pre-downloaded FinBERT
├── docker-compose.yml          # One-command startup
├── app/
│   ├── app.py                  # Landing page
│   ├── demo_utils.py           # Live model calls + fallbacks
│   └── pages/
│       ├── 0_Presentation.py   # 22-slide interactive deck
│       ├── 1_Numerical_Reasoning.py
│       ├── 2_Financial_Ratios.py
│       ├── 3_Sentiment_Analysis.py
│       ├── 4_Benchmark_Results.py
│       ├── 4_Benchmark_Results.py
│       └── 5_How_It_Works.py
├── src/                        # Core library (models, RAG, evaluation)
├── data/                       # Sample test cases
├── requirements.txt            # Full dependencies
└── requirements-demo.txt       # Slim deps for Docker
```

## Key Talking Points

1. **Fine-tuning teaches SKILLS, RAG provides INFORMATION**
   - FinBERT understands "headwinds" = negative (learned during training)
   - RAG can only look up similar examples - it doesn't truly understand

2. **Apples-to-apples: same task, different results**
   - FinBERT: 100% accuracy, 8ms, 90%+ confidence
   - RAG: 88% accuracy, 580ms, 33-67% confidence

3. **When to use RAG instead**
   - Data changes daily (news, market data)
   - Need source citations for compliance
   - No training data available

4. **The hybrid approach delivers best accuracy**
   - 65.8% vs 61.2% (fine-tuned) vs 15.3% (RAG only)

## Benchmark Results

| Metric | Fine-Tuned | RAG | Hybrid |
|--------|------------|-----|--------|
| FinQA Accuracy | 61.2% | 15.3% | **65.8%** |
| Sentiment Accuracy | 94.2% | 78.5% | N/A |
| Latency | ~200ms | ~800ms | ~450ms |
| Output Consistency | 98% | 65% | 95% |

## Models Used

| Model | Purpose |
|-------|---------|
| `ProsusAI/finbert` | Financial sentiment (fine-tuned BERT) |
| `truocpham/FinQA-7B-Instruct-v0.1` | Numerical reasoning (benchmark reference) |
| GPT-4o-mini (via API) | Base model & RAG comparison |

## License

MIT License
"# finetune-vs-rag" 
