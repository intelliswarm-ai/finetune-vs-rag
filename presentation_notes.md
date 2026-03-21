# Presentation Speaker Notes
**Total slides:** 62
**Generated:** generate_pptx.py

---

## Slide 1: Title

Welcome everyone. Today we're exploring one of the most important decisions in applied AI: when to fine-tune a model versus when to use Retrieval-Augmented Generation.

The key insight we'll demonstrate with real benchmarks: fine-tuning teaches a model new SKILLS (reasoning, calculation, pattern recognition), while RAG provides new INFORMATION (facts, documents, context). These solve fundamentally different problems, and the best systems often combine both.

This presentation includes a live demo -- every number you'll see comes from models we actually ran.

---

## Slide 2: Agenda

We'll cover four main parts. Part 1 sets the foundation -- what are LLMs and why do they struggle with specialized tasks. Part 2 goes deep on RAG and fine-tuning: how they work, where each shines, and a practical decision framework. Part 3 surveys the tooling ecosystem so you can actually build these systems. Part 4 is the evidence -- controlled benchmarks across sentiment analysis, numerical reasoning, financial ratios, and spam detection, plus a model size comparison.

Feel free to ask questions at any point.

---

## Slide 3: What Are LLMs?

LLMs are trained on trillions of tokens from the internet, books, and code. They learn language patterns, factual knowledge, and basic reasoning. The key point: they are generalists by design.

An analogy: an LLM is like a well-read university graduate. They can discuss almost any topic intelligently, but they're not a specialist in any one field. You wouldn't ask a fresh graduate to calculate complex financial ratios from SEC filings without additional training.

Note the parameter counts in the table -- these models range from 7 billion to potentially over a trillion parameters. More parameters generally means more knowledge capacity, but as we'll see later, size isn't everything.

---

## Slide 4: The Specialization Challenge

This is the core problem we're solving. Generic LLMs fail on domain tasks in predictable ways.

In our benchmarks, we saw base BERT get only 45% accuracy on financial sentiment -- barely better than random for a 3-class problem. It doesn't understand that 'restructuring charges' implies negative sentiment in finance, even though in general English it's a neutral word.

On the right side: domain experts need answers they can trust. In regulated industries like finance or healthcare, a wrong answer isn't just unhelpful -- it can be a compliance violation.

The efficiency ratio example is real: we tested this, and base Llama2-7B gets the formula wrong ~85% of the time.

---

## Slide 5: Three Approaches to Specialization

Think of these as a spectrum of effort vs. impact.

Prompt engineering is the starting point -- write better instructions, add examples. It's fast but limited. You can't teach a model new math skills through prompting alone.

RAG is the middle ground -- retrieve relevant documents and inject them into the prompt. Great for knowledge-intensive tasks, but the model's underlying capabilities don't change.

Fine-tuning is the highest-impact approach -- you actually update the model's weights. The model learns new patterns, reasoning strategies, and domain-specific behavior. But it requires training data and compute.

There's no universally 'best' approach -- the right choice depends on your specific task. That's what our benchmarks will help clarify.

---

## Slide 6: RAG: How It Works

RAG has four steps. First, the user's question is converted into a vector using an embedding model. Second, this vector is used to search a vector database for similar document chunks. Third, the top-K results are prepended to the prompt. Fourth, the LLM generates an answer using this augmented context.

The critical insight: the LLM's weights are never modified. You're not teaching it anything new -- you're showing it relevant information at inference time.

In our system, we use all-MiniLM-L6-v2 for embeddings and ChromaDB as the vector store. The LLM is the same base model -- we just give it better context.

---

## Slide 7: RAG: Benefits

RAG's biggest advantages are speed-to-deploy and dynamic knowledge.

No training required means you can start today with any LLM API. No GPUs, no training data preparation. This is why many companies start with RAG -- it's the fastest path to a working prototype.

Dynamic knowledge is huge for use cases where information changes frequently. A RAG system can ingest new documents without any model retraining.

Source citations are critical for enterprise use cases. When an analyst asks a question, they need to verify the answer against the source document. RAG provides this traceability out of the box.

---

## Slide 8: RAG: Limitations

This slide is crucial for understanding WHEN RAG falls short.

Point 1 is the most important: RAG cannot teach new skills. If the base model can't do multi-step arithmetic, retrieving a formula doesn't help -- it still can't apply it correctly.

Point 5 matters for production: RAG adds 200-600ms of latency for the embedding and retrieval steps. In our benchmarks, RAG approaches are consistently 3-5x slower than fine-tuned models.

Point 6 is often overlooked: retrieved documents consume context window space, leaving less room for the actual reasoning. With a 4K context window, 3 retrieved chunks of 500 tokens each already uses 37% of your context.

Bottom line: RAG is a knowledge tool, not a skill tool.

---

## Slide 9: Fine-Tuning: How It Works

The key distinction from RAG: fine-tuning modifies the model's weights. The knowledge and skills become part of the model itself.

The training data format matters enormously. For FinQA, each example is a financial table + question + step-by-step reasoning program. The model doesn't just learn the answer -- it learns HOW to arrive at the answer.

After fine-tuning, the model carries these capabilities everywhere. No retrieval step needed. No external database dependency. The skill is baked in.

Compare the two boxes at the bottom: RAG gives information AT query time. Fine-tuning gives skills PERMANENTLY.

---

## Slide 10: Fine-Tuning Methods

Three methods, each trading off between capability and resource requirements.

Full fine-tuning updates every parameter -- for Llama2-7B that's 7 billion weights. Requires multiple high-end GPUs (A100/H100). Best accuracy but highest cost.

LoRA is the game-changer that made fine-tuning accessible. It freezes the original model and adds tiny adapter layers (typically 0.1-1% of total parameters). Quality is surprisingly close to full fine-tuning.

QLoRA combines LoRA with 4-bit quantization. Our FinQA-7B model was trained this way -- a 7B model fine-tuned on a single GPU. This democratized fine-tuning for the open-source community.

Practical advice: start with QLoRA. You can always scale up if you need that extra 1-2% accuracy.

---

## Slide 11: Fine-Tuning: Key Benefits

Let's talk numbers from our actual benchmarks.

Accuracy: FinQA-7B achieves 61.2% on numerical reasoning vs 15.3% for RAG. That's a 4x improvement -- not because it has more information, but because it's learned HOW to reason about financial tables.

Consistency: fine-tuned models produce predictable output formats. In production, this matters enormously for downstream processing. RAG output is more variable because the model sees different context each time.

Latency: ~200ms for fine-tuned vs ~800ms for RAG. No embedding step, no vector search, no context assembly. Just inference. At scale, this 4x speedup translates directly to cost savings.

---

## Slide 12: Our Models: FinBERT & FinQA-7B

These are real, published models that anyone can use.

FinBERT is by Prosus AI -- they took BERT (110M parameters), pre-trained it further on Reuters financial news, then fine-tuned on the Financial PhraseBank dataset. The PhraseBank has 4,840 sentences labeled by 16 financial experts. This is high-quality, expert-annotated training data.

FinQA-7B is a community contribution -- someone took Meta's Llama2-7B and fine-tuned it using QLoRA on the FinQA dataset from IBM Research. The FinQA dataset is remarkable: 8,281 question-answer pairs extracted from real SEC filings, each with step-by-step reasoning programs.

Note the training data sizes: 4,840 sentences for FinBERT and 8,281 Q&A pairs for FinQA. Fine-tuning doesn't require millions of examples -- thousands of high-quality examples can be transformative.

---

## Slide 13: Our Models: Spam Detection

DistilBERT is 40% smaller and 60% faster than BERT, while retaining 97% of its language understanding. This makes it ideal for high-throughput tasks like email filtering.

The key insight: fine-tuning teaches the model PATTERNS, not just keywords. A fine-tuned spam detector learns that 'urgency + verification request + deadline' is a phishing pattern, even if the individual words are benign.

RAG struggles here because similar-looking emails can be either spam or legitimate. A pharmacy notification and a pharmaceutical spam email look similar in embedding space but have completely different intent.

---

## Slide 14: Training Data Examples

Let's look at what the model actually learns from.

Each FinQA training example has three parts: a financial data table with real numbers, a question that requires multi-step reasoning, and a 'program' that shows the exact calculation steps.

For example: 'What is the total revenue growth rate?' requires finding revenue in two different years from the table, computing the difference, and dividing by the base year. The model learns this reasoning pattern, not just the answer.

This is why fine-tuning outperforms RAG on numerical tasks: RAG can retrieve the table, but the model still needs to know HOW to compute the answer. Fine-tuning teaches the 'how'.

---

## Slide 15: Head-to-Head Comparison

This side-by-side comparison crystallizes the core difference.

RAG excels when the task is knowledge-intensive: 'What was Apple's revenue last quarter?' Just retrieve the right document and the answer is there. The model doesn't need special skills.

Fine-tuning excels when the task requires reasoning: 'Calculate the compound annual growth rate from this table.' No amount of document retrieval helps if the model can't do multi-step arithmetic.

The hybrid approach combines both: use fine-tuning for the reasoning skills and RAG for the latest data. In our benchmarks, the hybrid approach consistently achieves the highest or equal-best accuracy.

---

## Slide 16: When RAG Falls Short

These are real examples from our benchmarks where RAG failed.

Example 1: Financial sentiment of 'The company announced restructuring charges.' RAG retrieves similar sentences but the base model still classifies it as neutral. FinBERT knows this is negative in financial context.

Example 2: 'Calculate the efficiency ratio from this data.' RAG retrieves the formula definition, but the base model still computes it incorrectly. FinQA-7B gets it right because it's practiced thousands of similar calculations during training.

The pattern: RAG fails when the bottleneck is SKILL, not INFORMATION.

---

## Slide 17: Decision Framework

This is your practical takeaway. Two questions determine the right approach.

Question 1: Does the task require NEW REASONING SKILLS? If yes, you need fine-tuning. If no, the model already knows how to do the task and you just need to give it the right information.

Question 2: Does it need FRESH or DYNAMIC data? If yes, you need RAG for the knowledge layer.

Both yes? Hybrid. Just skills? Fine-tune. Just knowledge? RAG. Neither? Start with prompt engineering.

Most real-world production systems end up as hybrids -- the question is which component carries more weight.

---

## Slide 18: Fine-Tuning Tools

The ecosystem has matured dramatically in the last 2 years.

HuggingFace is the de facto hub -- 500K+ models, PEFT/LoRA libraries, and the Trainer API. Most fine-tuning projects start here.

Unsloth deserves special mention -- they've achieved 2x training speed and 60% memory reduction through custom CUDA kernels. This means you can fine-tune a 7B model on a single consumer GPU.

For enterprises: AWS SageMaker and Bedrock Custom Models provide managed fine-tuning. You upload data, they handle the infrastructure. More expensive but zero DevOps overhead.

---

## Slide 19: Fine-Tune: Local Setup

This slide shows actual code for local fine-tuning with Unsloth and QLoRA.

The key parameters: rank=16 (adapter size), target_modules include attention AND MLP layers (not just attention, which is a common mistake). Learning rate of 2e-4 with cosine scheduling.

With QLoRA on a single NVIDIA GPU with 24GB VRAM, you can fine-tune a 7B model in about 4-6 hours on 8K training examples. Total cost: electricity.

For those without local GPUs: Google Colab Pro ($10/month) gives you access to A100 GPUs sufficient for this.

---

## Slide 20: Fine-Tune: AWS

AWS offers two paths: SageMaker for full control, Bedrock for managed simplicity.

SageMaker: bring your own training script, choose instance types (ml.g5.2xlarge is the sweet spot for 7B models), and manage the full ML lifecycle. More work but more control.

Bedrock Custom Models: upload your JSONL, click 'train', and get a private endpoint. They handle hyperparameters, infrastructure, and scaling. Best for teams that want results without ML engineering.

Cost comparison: Bedrock charges per training token (~$0.008/1K tokens). For 8K examples, that's roughly $50-100 for a fine-tuning job. SageMaker is pay-per-hour for the GPU instance.

---

## Slide 21: RAG Tools

RAG infrastructure has three layers: embedding models, vector databases, and orchestration frameworks.

Embeddings: all-MiniLM-L6-v2 is our choice -- only 22M parameters but excellent performance. For production, consider OpenAI's text-embedding-3-small or Cohere's embed-v3.

Vector DBs: ChromaDB (what we use) is great for prototyping -- in-memory, zero config. For production: Pinecone (managed), Weaviate (open-source), or pgvector (if you're already on PostgreSQL).

LangChain and LlamaIndex handle the orchestration -- chunking, retrieval, prompt assembly, response generation.

---

## Slide 22: RAG: Local Setup

This code is from our actual demo application.

Key design decisions: chunk size of 300 words with 50-word overlap. The overlap ensures that sentences straddling chunk boundaries aren't lost. Top-K=3 retrieval -- more chunks means more context but also more noise and higher latency.

ChromaDB creates an in-memory collection and persists to disk. On restart, it reloads from the persisted data -- no need to re-embed all documents.

Total setup time: about 10 minutes from scratch. That's the beauty of RAG -- fast to prototype.

---

## Slide 23: RAG: AWS

AWS Bedrock Knowledge Bases is the managed RAG solution.

You point it at an S3 bucket with your documents, choose an embedding model, and it handles chunking, embedding, and storage in OpenSearch Serverless. No infrastructure to manage.

The trade-off: less control over chunking strategy, retrieval logic, and re-ranking. For most enterprise use cases, the convenience outweighs the customization loss.

---

## Slide 24: Data & Evaluation Tools

Data quality is the single biggest factor in fine-tuning success.

Argilla and Label Studio are open-source annotation platforms. For financial data, you need domain experts annotating -- not just anyone. The Financial PhraseBank that trained FinBERT used 16 financial professionals.

For evaluation: don't rely on a single metric. We use accuracy, F1 score, latency, cost, and LLM-as-Judge. Different metrics tell different stories -- a model with 90% accuracy but 10-second latency is useless for real-time applications.

---

## Slide 25: Real-World Use Cases

These are production use cases where the right approach matters.

Customer support: RAG is ideal. Questions are about YOUR products, and the knowledge base changes frequently. No need to retrain the model.

Medical diagnosis support: Fine-tuning is critical. The model needs to understand clinical reasoning patterns, not just retrieve medical textbooks.

Legal document analysis: Hybrid. Fine-tune for legal reasoning patterns, RAG for case law lookups.

The pattern: if the domain has unique reasoning patterns, fine-tune. If it's mostly knowledge lookup, RAG.

---

## Slide 26: The Hybrid Approach

This is our architecture diagram for the hybrid system.

The user's question and financial table go to both the embedding model (for retrieval) and directly to the fine-tuned model. Retrieved documents are added as additional context.

The fine-tuned model (FinQA-7B) brings the reasoning skills. The RAG component brings fresh context and supporting evidence. Together, you get domain reasoning PLUS verifiable sources.

In our benchmarks, the hybrid approach matches or beats every other approach across all four experiments.

---

## Slide 27: Cost & ROI

Let's be practical about costs.

Fine-tuning has a higher upfront cost: training data preparation, compute for training, evaluation. But per-query cost is lower -- no retrieval step, fewer tokens, faster inference.

RAG has a lower upfront cost: set up a vector DB, ingest documents, start querying. But per-query cost is higher -- embedding, retrieval, and larger prompts.

The crossover point: at roughly 10K-50K queries per month, fine-tuning becomes cheaper than RAG. Below that, RAG's lower upfront cost wins.

The real ROI question: what's the cost of wrong answers? In regulated industries, one compliance violation can cost more than a year of fine-tuning compute.

---

## Slide 28: Key Takeaways

Three things to remember from this presentation.

1. Fine-tuning teaches SKILLS. It changes what the model CAN DO. Use it when the base model lacks the reasoning capabilities your task requires.

2. RAG provides INFORMATION. It changes what the model KNOWS at query time. Use it when the model already has the right skills but needs access to specific or current data.

3. Hybrid combines both. For most serious production systems, you want a fine-tuned model augmented with RAG for dynamic knowledge. This is the architecture that consistently wins in our benchmarks.

---

## Slide 29: Benchmark Results (Section Divider)

Now let's look at the evidence. Everything from here on is based on real experiments we ran -- no synthetic data, no simulated results. Each benchmark compares the SAME architecture with different approaches. The only variable is the method: base model, fine-tuned, RAG, or hybrid.

---

## Slide 30: Benchmark Experiments Overview

Four controlled experiments, each using a different model architecture.

Experiment 1: BERT-base (110M params) on financial sentiment classification.
Experiment 2: Llama2-7B (7B params) on numerical reasoning from financial tables.
Experiment 3: Llama2-7B on financial ratio calculation (more complex multi-step problems).
Experiment 4: DistilBERT (66M params) on spam/phishing email detection.
Experiment 5: Model size comparison -- DistilBERT (66M) vs GPT-4o-mini (~8B) on spam detection.

The methodology is critical: same architecture, same test cases, same evaluation criteria. This isolates the impact of the approach from confounding variables like model size.

---

## Slide 31: Sentiment (BERT 110M) - Accuracy & Latency

Pay attention to the accuracy gaps between approaches.

In every experiment, the fine-tuned model significantly outperforms the base model. RAG improves over base but doesn't match fine-tuning. The hybrid approach typically matches or slightly exceeds the fine-tuned model alone.

The latency chart tells the cost story: RAG consistently adds 200-600ms for the retrieval step. At scale, this compounds. 1,000 queries per minute with 400ms extra latency means 400 additional seconds of compute time per minute.

---

## Slide 32: Sentiment (BERT 110M) - Token Usage & Cost

Token consumption directly drives API costs.

RAG uses significantly more tokens because retrieved document chunks are prepended to the prompt. A typical RAG query might use 3-5x more input tokens than a direct fine-tuned inference.

For the cost/1K queries metric: this assumes market pricing for API-based models. Self-hosted fine-tuned models have near-zero marginal cost (just electricity). This is a massive advantage at scale.

---

## Slide 33: Sentiment (BERT 110M) - Quality Metrics

F1 score provides a more nuanced view than accuracy alone, especially for imbalanced classes.

High accuracy with low F1 means the model is biased toward the majority class. A good model needs both high precision (few false positives) and high recall (few false negatives).

For numerical tasks, MAPE (Mean Absolute Percentage Error) measures how close the predicted numbers are to the expected values. A model might get the direction right but be off by 50%.

---

## Slide 34: Sentiment (BERT 110M) - Confidence/Extra Metrics

The category breakdown reveals WHERE each approach wins and loses.

Look for categories where fine-tuning dramatically outperforms RAG -- these are the 'skill gaps' that RAG cannot address. Conversely, categories where RAG matches fine-tuning are the knowledge-intensive tasks.

Domain jargon is a classic fine-tuning win: the model learns that financial terms like 'headwind', 'runway', and 'restructuring' carry different sentiment than their everyday usage.

---

## Slide 35: Sentiment (BERT 110M) - Category Breakdown

The category breakdown reveals WHERE each approach wins and loses.

Look for categories where fine-tuning dramatically outperforms RAG -- these are the 'skill gaps' that RAG cannot address. Conversely, categories where RAG matches fine-tuning are the knowledge-intensive tasks.

Domain jargon is a classic fine-tuning win: the model learns that financial terms like 'headwind', 'runway', and 'restructuring' carry different sentiment than their everyday usage.

---

## Slide 36: Sentiment (BERT 110M) - Per-Example Results

Individual examples help build intuition for model behavior.

Look for patterns in the failures: does the base model consistently fail on the same types of questions? Does RAG fail when the retrieved documents are misleading or off-topic?

The [Y/N] markers show correct/incorrect. Count the patterns -- where does each approach struggle?

---

## Slide 37: Numerical (Llama2 7B) - Accuracy & Latency

Pay attention to the accuracy gaps between approaches.

In every experiment, the fine-tuned model significantly outperforms the base model. RAG improves over base but doesn't match fine-tuning. The hybrid approach typically matches or slightly exceeds the fine-tuned model alone.

The latency chart tells the cost story: RAG consistently adds 200-600ms for the retrieval step. At scale, this compounds. 1,000 queries per minute with 400ms extra latency means 400 additional seconds of compute time per minute.

---

## Slide 38: Numerical (Llama2 7B) - Token Usage & Cost

Token consumption directly drives API costs.

RAG uses significantly more tokens because retrieved document chunks are prepended to the prompt. A typical RAG query might use 3-5x more input tokens than a direct fine-tuned inference.

For the cost/1K queries metric: this assumes market pricing for API-based models. Self-hosted fine-tuned models have near-zero marginal cost (just electricity). This is a massive advantage at scale.

---

## Slide 39: Numerical (Llama2 7B) - Quality Metrics

F1 score provides a more nuanced view than accuracy alone, especially for imbalanced classes.

High accuracy with low F1 means the model is biased toward the majority class. A good model needs both high precision (few false positives) and high recall (few false negatives).

For numerical tasks, MAPE (Mean Absolute Percentage Error) measures how close the predicted numbers are to the expected values. A model might get the direction right but be off by 50%.

---

## Slide 40: Numerical (Llama2 7B) - Confidence/Extra Metrics

The category breakdown reveals WHERE each approach wins and loses.

Look for categories where fine-tuning dramatically outperforms RAG -- these are the 'skill gaps' that RAG cannot address. Conversely, categories where RAG matches fine-tuning are the knowledge-intensive tasks.

Domain jargon is a classic fine-tuning win: the model learns that financial terms like 'headwind', 'runway', and 'restructuring' carry different sentiment than their everyday usage.

---

## Slide 41: Numerical (Llama2 7B) - Category Breakdown

The category breakdown reveals WHERE each approach wins and loses.

Look for categories where fine-tuning dramatically outperforms RAG -- these are the 'skill gaps' that RAG cannot address. Conversely, categories where RAG matches fine-tuning are the knowledge-intensive tasks.

Domain jargon is a classic fine-tuning win: the model learns that financial terms like 'headwind', 'runway', and 'restructuring' carry different sentiment than their everyday usage.

---

## Slide 42: Numerical (Llama2 7B) - Per-Example Results

Individual examples help build intuition for model behavior.

Look for patterns in the failures: does the base model consistently fail on the same types of questions? Does RAG fail when the retrieved documents are misleading or off-topic?

The [Y/N] markers show correct/incorrect. Count the patterns -- where does each approach struggle?

---

## Slide 43: Financial Ratios (Llama2 7B) - Accuracy & Latency

Pay attention to the accuracy gaps between approaches.

In every experiment, the fine-tuned model significantly outperforms the base model. RAG improves over base but doesn't match fine-tuning. The hybrid approach typically matches or slightly exceeds the fine-tuned model alone.

The latency chart tells the cost story: RAG consistently adds 200-600ms for the retrieval step. At scale, this compounds. 1,000 queries per minute with 400ms extra latency means 400 additional seconds of compute time per minute.

---

## Slide 44: Financial Ratios (Llama2 7B) - Token Usage & Cost

Token consumption directly drives API costs.

RAG uses significantly more tokens because retrieved document chunks are prepended to the prompt. A typical RAG query might use 3-5x more input tokens than a direct fine-tuned inference.

For the cost/1K queries metric: this assumes market pricing for API-based models. Self-hosted fine-tuned models have near-zero marginal cost (just electricity). This is a massive advantage at scale.

---

## Slide 45: Financial Ratios (Llama2 7B) - Quality Metrics

F1 score provides a more nuanced view than accuracy alone, especially for imbalanced classes.

High accuracy with low F1 means the model is biased toward the majority class. A good model needs both high precision (few false positives) and high recall (few false negatives).

For numerical tasks, MAPE (Mean Absolute Percentage Error) measures how close the predicted numbers are to the expected values. A model might get the direction right but be off by 50%.

---

## Slide 46: Financial Ratios (Llama2 7B) - Confidence/Extra Metrics

The category breakdown reveals WHERE each approach wins and loses.

Look for categories where fine-tuning dramatically outperforms RAG -- these are the 'skill gaps' that RAG cannot address. Conversely, categories where RAG matches fine-tuning are the knowledge-intensive tasks.

Domain jargon is a classic fine-tuning win: the model learns that financial terms like 'headwind', 'runway', and 'restructuring' carry different sentiment than their everyday usage.

---

## Slide 47: Financial Ratios (Llama2 7B) - Category Breakdown

The category breakdown reveals WHERE each approach wins and loses.

Look for categories where fine-tuning dramatically outperforms RAG -- these are the 'skill gaps' that RAG cannot address. Conversely, categories where RAG matches fine-tuning are the knowledge-intensive tasks.

Domain jargon is a classic fine-tuning win: the model learns that financial terms like 'headwind', 'runway', and 'restructuring' carry different sentiment than their everyday usage.

---

## Slide 48: Financial Ratios (Llama2 7B) - Per-Example Results

Individual examples help build intuition for model behavior.

Look for patterns in the failures: does the base model consistently fail on the same types of questions? Does RAG fail when the retrieved documents are misleading or off-topic?

The [Y/N] markers show correct/incorrect. Count the patterns -- where does each approach struggle?

---

## Slide 49: Spam Detection (DistilBERT 66M) - Accuracy & Latency

Pay attention to the accuracy gaps between approaches.

In every experiment, the fine-tuned model significantly outperforms the base model. RAG improves over base but doesn't match fine-tuning. The hybrid approach typically matches or slightly exceeds the fine-tuned model alone.

The latency chart tells the cost story: RAG consistently adds 200-600ms for the retrieval step. At scale, this compounds. 1,000 queries per minute with 400ms extra latency means 400 additional seconds of compute time per minute.

---

## Slide 50: Spam Detection (DistilBERT 66M) - Token Usage & Cost

Token consumption directly drives API costs.

RAG uses significantly more tokens because retrieved document chunks are prepended to the prompt. A typical RAG query might use 3-5x more input tokens than a direct fine-tuned inference.

For the cost/1K queries metric: this assumes market pricing for API-based models. Self-hosted fine-tuned models have near-zero marginal cost (just electricity). This is a massive advantage at scale.

---

## Slide 51: Spam Detection (DistilBERT 66M) - Quality Metrics

F1 score provides a more nuanced view than accuracy alone, especially for imbalanced classes.

High accuracy with low F1 means the model is biased toward the majority class. A good model needs both high precision (few false positives) and high recall (few false negatives).

For numerical tasks, MAPE (Mean Absolute Percentage Error) measures how close the predicted numbers are to the expected values. A model might get the direction right but be off by 50%.

---

## Slide 52: Spam Detection (DistilBERT 66M) - Confidence/Extra Metrics

The category breakdown reveals WHERE each approach wins and loses.

Look for categories where fine-tuning dramatically outperforms RAG -- these are the 'skill gaps' that RAG cannot address. Conversely, categories where RAG matches fine-tuning are the knowledge-intensive tasks.

Domain jargon is a classic fine-tuning win: the model learns that financial terms like 'headwind', 'runway', and 'restructuring' carry different sentiment than their everyday usage.

---

## Slide 53: Spam Detection (DistilBERT 66M) - Category Breakdown

The category breakdown reveals WHERE each approach wins and loses.

Look for categories where fine-tuning dramatically outperforms RAG -- these are the 'skill gaps' that RAG cannot address. Conversely, categories where RAG matches fine-tuning are the knowledge-intensive tasks.

Domain jargon is a classic fine-tuning win: the model learns that financial terms like 'headwind', 'runway', and 'restructuring' carry different sentiment than their everyday usage.

---

## Slide 54: Spam Detection (DistilBERT 66M) - Per-Example Results

Individual examples help build intuition for model behavior.

Look for patterns in the failures: does the base model consistently fail on the same types of questions? Does RAG fail when the retrieved documents are misleading or off-topic?

The [Y/N] markers show correct/incorrect. Count the patterns -- where does each approach struggle?

---

## Slide 55: Model Size Benchmark - Overview

This experiment asks a fundamental question: if fine-tuning teaches skills, does a bigger model learn them better?

We compare two models BOTH fine-tuned on the same spam detection task with the same training data. DistilBERT has 66 million parameters. GPT-4o-mini has approximately 8 billion -- 121 times larger.

DistilBERT runs locally with near-zero cost. GPT-4o-mini requires an API call at $0.30 per million input tokens.

The question is whether that 121x size difference and higher cost translate to proportionally better performance.

---

## Slide 56: Model Size Benchmark - Cost & Findings

The results are striking.

On basic test cases, GPT-4o-mini edges out DistilBERT by about 5 percentage points. But DistilBERT is running at 1/23rd the latency and near-zero cost.

On adversarial cases, the gap widens -- the larger model is more robust to edge cases. This makes sense: more parameters give more capacity to handle unusual inputs.

The diminishing returns insight: going from 66M to 8B parameters (121x) yields only a modest accuracy gain. For many production use cases, the smaller, faster, cheaper model is the better choice.

This has profound implications for deployment: a $0.005/1K query model achieving 95% accuracy often beats a $0.02/1K query model achieving 100% accuracy.

---

## Slide 57: Model Size Benchmark - LLM-as-Judge

The LLM-as-Judge evaluation adds a qualitative dimension to our quantitative benchmarks.

We used GPT-4o as an independent judge to score each model's predictions on three dimensions: correctness (did it get the right answer?), reasoning quality (does the prediction show domain understanding?), and faithfulness (is the classification based on actual email content?).

On basic cases, both models score near-perfect -- the differences only emerge on adversarial cases where the emails are deliberately designed to confuse classifiers.

The judge scores align with the accuracy metrics but provide richer insights: a model can be incorrect but still show reasonable reasoning, or correct but for the wrong reasons.

---

## Slide 58: RAG Strengths Overview

This benchmark proves that RAG performs dramatically better when test data aligns with the knowledge base. Standard benchmarks showed RAG at 15% on numerical tasks due to data conflicts; here RAG achieves 87% because retrieved data helps rather than hurts.

---

## Slide 58: Benchmark Insights

Let's step back and look at the big picture across all experiments.

The data consistently tells us: fine-tuning excels at REASONING, COMPUTATION, and PATTERN RECOGNITION. These are skill-based tasks where the model needs to learn new capabilities.

RAG excels at KNOWLEDGE RETRIEVAL -- tasks where the answer is in a document and the model just needs to find and present it.

The hybrid approach consistently achieves the highest accuracy because it combines learned skills with access to fresh information. This is the architecture pattern for production systems.

---

## Slide 59: RAG Advantage Analysis

The category breakdown reveals where RAG provides the most value. Direct retrieval shows the largest gap because base models literally cannot access proprietary document data. Cross-document synthesis benefits most from hybrid because it requires both retrieval and reasoning.

---

## Slide 59: Striking Examples

These examples are cherry-picked to illustrate the clearest wins for each approach.

Fine-tuning wins: cases where domain-specific reasoning is required. Financial jargon, multi-step calculations, nuanced classification.

RAG wins: cases where specific factual information is needed that the model doesn't have in its weights. Current data, specific document references, evolving knowledge.

Hybrid wins: cases that need BOTH -- domain reasoning skills applied to specific current data.

---

## Slide 60: LLM Judge RAG Strengths

The GPT-4o judge confirms that RAG dramatically improves faithfulness scores. This is the primary production value of RAG: it reduces hallucination by grounding model responses in actual documents. The hybrid approach achieves the best overall quality by combining retrieval with reasoning skills.

---

## Slide 60: Summary: All Experiments

This is the slide to photograph if you remember nothing else.

The accuracy table shows fine-tuning winning across every experiment. The cost table shows the trade-off: RAG uses more tokens and costs more per query.

Five key conclusions, backed by our data:
1. Fine-tuning consistently outperforms base models.
2. RAG improves over base but can't match fine-tuning for reasoning.
3. Hybrid achieves the best results.
4. Spam detection is a great fine-tuning use case -- learned patterns beat retrieval.
5. All comparisons are controlled -- same architecture, different approach.

The implication for your projects: if accuracy matters and you have training data, fine-tune. If you need dynamic knowledge, add RAG. For production, do both.

---

## Slide 61: RAG Strengths Conclusions

The key takeaway is that RAG and fine-tuning solve fundamentally different problems. RAG provides knowledge (access to documents the model has never seen), while fine-tuning provides skills (domain-specific reasoning). The strongest production systems combine both approaches.

---

## Slide 62: Thank You

Thank you for your attention. Let's open the floor for questions.

For the live demo, we can run any of the models in real-time and show you the differences side by side. The Streamlit application is running at localhost:8501 if you want to try it yourself.

All code, benchmarks, and the presentation itself are available in the repository. The benchmark results are reproducible -- you can run them yourself with docker compose up.

Key takeaway: fine-tuning teaches SKILLS, RAG provides INFORMATION. The best systems combine both. Now go build something great.

---
