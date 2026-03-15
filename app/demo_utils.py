"""
Live Demo Utilities -- Real Models Only
- FinBERT: local fine-tuned sentiment model (transformers)
- Mistral: local LLM via Ollama for base / RAG / hybrid generation
- RAG: sentence-transformers + ChromaDB with real financial documents
No simulated or mocked responses.
"""
import os
import time
from typing import Optional, List, Dict, Tuple, Generator
from dataclasses import dataclass
from pathlib import Path

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

LLM_MODEL = os.getenv("LLM_MODEL", "llama2")
FINETUNED_LLM_MODEL = os.getenv("FINETUNED_LLM_MODEL", "finqa-7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Response data classes
# ---------------------------------------------------------------------------
@dataclass
class SentimentResult:
    label: str
    confidence: float
    scores: Dict[str, float]
    latency_ms: float
    model_name: str
    is_live: bool = True


@dataclass
class LLMResponse:
    answer: str
    latency_ms: float
    model_name: str
    reasoning_steps: Optional[str] = None
    retrieved_context: Optional[List[str]] = None
    is_live: bool = True


@dataclass
class RAGSentimentResult:
    label: str
    confidence: float
    scores: Dict[str, float]
    latency_ms: float
    model_name: str
    retrieved_examples: List[Dict[str, str]]
    retrieval_ms: float
    generation_ms: float
    is_live: bool = True


# ---------------------------------------------------------------------------
# Ollama / LLM client
# ---------------------------------------------------------------------------
_llm_client = None
_llm_checked = False


def _get_llm_client():
    """Return an OpenAI-compatible client pointing at Ollama."""
    global _llm_client, _llm_checked
    if _llm_checked:
        return _llm_client
    _llm_checked = True
    try:
        from openai import OpenAI
        _llm_client = OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")
        # Quick connectivity check
        _llm_client.models.list()
        return _llm_client
    except Exception:
        _llm_client = None
        return None


def _reset_llm_client():
    """Force re-check on next call (useful after Ollama starts)."""
    global _llm_checked
    _llm_checked = False


def has_llm() -> bool:
    return _get_llm_client() is not None


def _llm_error_message() -> str:
    return (
        f"[Ollama not reachable at {OLLAMA_HOST}]\n\n"
        "Start Ollama and pull a model:\n"
        "```\nollama serve\n"
        f"ollama pull {LLM_MODEL}\n```\n\n"
        "Or with Docker Compose: `docker compose up`"
    )


# ---------------------------------------------------------------------------
# RAG engine (lazy singleton)
# ---------------------------------------------------------------------------
_rag_engine = None


def _get_rag():
    global _rag_engine
    if _rag_engine is None:
        from rag_engine import RAGEngine
        _rag_engine = RAGEngine.get_instance()
        if not _rag_engine.is_ready:
            _rag_engine.initialize()
    return _rag_engine


def rag_ready() -> bool:
    try:
        return _get_rag().is_ready
    except Exception:
        return False


def rag_num_chunks() -> int:
    try:
        return _get_rag().num_chunks
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# FinBERT (local fine-tuned model for sentiment)
# ---------------------------------------------------------------------------
_finbert_model = None
_finbert_tokenizer = None
_finbert_loaded = False
_finbert_load_attempted = False


def _load_finbert():
    global _finbert_model, _finbert_tokenizer, _finbert_loaded, _finbert_load_attempted
    if _finbert_load_attempted:
        return _finbert_loaded
    _finbert_load_attempted = True
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        model_id = "ProsusAI/finbert"
        _finbert_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        _finbert_model.eval()
        _finbert_loaded = True
        return True
    except Exception as e:
        print(f"[demo_utils] Could not load FinBERT: {e}")
        return False


def run_finbert(text: str) -> SentimentResult:
    """Run real FinBERT inference. Raises if model unavailable."""
    if not _load_finbert():
        raise RuntimeError(
            "FinBERT model not loaded. Install torch and transformers."
        )
    import torch
    start = time.perf_counter()
    inputs = _finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _finbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    labels = ["positive", "negative", "neutral"]
    scores = {label: round(probs[i].item(), 4) for i, label in enumerate(labels)}
    best_idx = probs.argmax().item()

    return SentimentResult(
        label=labels[best_idx],
        confidence=round(probs[best_idx].item(), 4),
        scores=scores,
        latency_ms=round(elapsed_ms, 1),
        model_name="ProsusAI/finbert (local)",
        is_live=True,
    )


def finbert_available() -> bool:
    return _load_finbert()


# ---------------------------------------------------------------------------
# Base BERT sentiment (same 110M architecture, NO fine-tuning, NO RAG)
# ---------------------------------------------------------------------------
# Simple label descriptions used as "prototypes" for zero-shot similarity
_BASE_LABEL_PROTOTYPES = {
    "positive": "good results, profit growth, revenue increase, strong performance",
    "negative": "bad results, loss, decline, poor performance, weakness",
    "neutral": "unchanged, stable, routine, no significant change",
}
_proto_embeddings = None


def run_base_bert_sentiment(text: str) -> SentimentResult:
    """Classify sentiment using bert-base-uncased WITHOUT fine-tuning or RAG.

    Uses cosine similarity between the text embedding and simple label
    prototype embeddings. This is the most basic thing you can do with a
    base model -- no training, no retrieval, just raw embeddings.
    """
    import torch
    if not _load_bert_base():
        raise RuntimeError("bert-base-uncased not loaded")

    global _proto_embeddings
    if _proto_embeddings is None:
        _proto_embeddings = _embed_texts_bert(
            list(_BASE_LABEL_PROTOTYPES.values())
        )

    start = time.perf_counter()
    query_emb = _embed_texts_bert([text])
    similarities = torch.mm(query_emb, _proto_embeddings.t())[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    labels = list(_BASE_LABEL_PROTOTYPES.keys())
    scores_raw = {labels[i]: similarities[i].item() for i in range(len(labels))}

    # Softmax to get probabilities
    sim_tensor = similarities
    probs = torch.nn.functional.softmax(sim_tensor * 5, dim=-1)  # temperature=0.2
    scores = {labels[i]: round(probs[i].item(), 4) for i in range(len(labels))}
    best_idx = probs.argmax().item()

    return SentimentResult(
        label=labels[best_idx],
        confidence=round(probs[best_idx].item(), 4),
        scores=scores,
        latency_ms=round(elapsed_ms, 1),
        model_name="bert-base-uncased (110M, no fine-tuning, no RAG)",
        is_live=True,
    )


# ---------------------------------------------------------------------------
# RAG-based sentiment using BERT-base (same 110M architecture as FinBERT)
# ---------------------------------------------------------------------------
# Labeled examples the RAG retrieves from (the "knowledge base")
SENTIMENT_KNOWLEDGE_BASE = [
    {"text": "Revenue exceeded analyst expectations by 8%.", "label": "positive"},
    {"text": "Quarterly profit margins expanded to a five-year high.", "label": "positive"},
    {"text": "Loan growth accelerated across all segments.", "label": "positive"},
    {"text": "Customer acquisition reached record levels in Q4.", "label": "positive"},
    {"text": "Net income rose 22% driven by higher interest rates.", "label": "positive"},
    {"text": "The firm reported a net loss for the third consecutive quarter.", "label": "negative"},
    {"text": "Operating expenses surged 20% due to regulatory fines.", "label": "negative"},
    {"text": "Credit quality deteriorated amid rising delinquencies.", "label": "negative"},
    {"text": "The bank warned of material weakness in internal controls.", "label": "negative"},
    {"text": "Provisions for bad debts increased sharply on weaker outlook.", "label": "negative"},
    {"text": "Total deposits remained flat compared to the prior period.", "label": "neutral"},
    {"text": "The board approved a routine extension of the credit facility.", "label": "neutral"},
    {"text": "Staffing levels were unchanged from last quarter.", "label": "neutral"},
    {"text": "The company filed its annual 10-K report on time.", "label": "neutral"},
    {"text": "Total branch count held steady at 4,200 locations nationwide.", "label": "neutral"},
]

_bert_base_model = None
_bert_base_tokenizer = None
_bert_base_loaded = False
_bert_base_load_attempted = False
_kb_embeddings = None


def _load_bert_base():
    """Load bert-base-uncased (same architecture as FinBERT, NOT fine-tuned)."""
    global _bert_base_model, _bert_base_tokenizer, _bert_base_loaded
    global _bert_base_load_attempted, _kb_embeddings
    if _bert_base_load_attempted:
        return _bert_base_loaded
    _bert_base_load_attempted = True
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        model_id = "bert-base-uncased"
        _bert_base_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _bert_base_model = AutoModel.from_pretrained(model_id)
        _bert_base_model.eval()

        # Pre-embed the knowledge base
        _kb_embeddings = _embed_texts_bert(
            [ex["text"] for ex in SENTIMENT_KNOWLEDGE_BASE]
        )
        _bert_base_loaded = True
        print("[demo_utils] bert-base-uncased loaded for RAG comparison")
        return True
    except Exception as e:
        print(f"[demo_utils] Could not load bert-base-uncased: {e}")
        return False


def _embed_texts_bert(texts: List[str]):
    """Get [CLS] embeddings from bert-base-uncased."""
    import torch
    inputs = _bert_base_tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=512, padding=True,
    )
    with torch.no_grad():
        outputs = _bert_base_model(**inputs)
    # Use [CLS] token embedding (first token)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    # L2 normalise for cosine similarity
    return torch.nn.functional.normalize(cls_embeddings, dim=-1)


def bert_base_available() -> bool:
    return _load_bert_base()


def run_hybrid_sentiment(text: str) -> SentimentResult:
    """Hybrid sentiment: FinBERT scores + RAG retrieval combined.

    Apples-to-apples: same 110M BERT architecture.
    Combines fine-tuned classification (FinBERT) with RAG retrieval
    (bert-base similarity voting), weighted 60/40 toward fine-tuning.
    """
    import torch
    start = time.perf_counter()

    # Get FinBERT prediction
    finbert_result = run_finbert(text)
    finbert_scores = finbert_result.scores  # {"positive": ..., "negative": ..., "neutral": ...}

    # Get RAG prediction
    rag_result = run_rag_sentiment(text)
    rag_scores = rag_result.scores

    # Combine: weighted blend (60% fine-tuned, 40% RAG)
    ft_weight, rag_weight = 0.6, 0.4
    combined = {}
    for label in ["positive", "negative", "neutral"]:
        combined[label] = round(
            ft_weight * finbert_scores.get(label, 0) +
            rag_weight * rag_scores.get(label, 0),
            4,
        )

    best_label = max(combined, key=combined.get)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return SentimentResult(
        label=best_label,
        confidence=combined[best_label],
        scores=combined,
        latency_ms=round(elapsed_ms, 1),
        model_name="FinBERT + RAG hybrid (110M)",
        is_live=True,
    )


def run_rag_sentiment(text: str) -> RAGSentimentResult:
    """RAG sentiment using bert-base-uncased (110M, NOT fine-tuned).

    Apples-to-apples comparison with FinBERT:
    - Same base architecture: BERT-base (110M params)
    - Same input: financial text
    - Same output: positive / negative / neutral
    - Different approach: RAG retrieval + similarity voting vs fine-tuned classification
    """
    import torch
    start = time.perf_counter()

    if not _load_bert_base():
        raise RuntimeError("bert-base-uncased not loaded")

    # Step 1: Embed the query with bert-base-uncased
    retrieval_start = time.perf_counter()
    query_emb = _embed_texts_bert([text])

    # Step 2: Compute cosine similarity against knowledge base
    similarities = torch.mm(query_emb, _kb_embeddings.t())[0]
    top_k = 5
    top_indices = similarities.argsort(descending=True)[:top_k].tolist()
    top_examples = [SENTIMENT_KNOWLEDGE_BASE[i] for i in top_indices]
    top_sims = [similarities[i].item() for i in top_indices]
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    # Step 3: Similarity-weighted vote
    generation_start = time.perf_counter()
    votes = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for ex, sim in zip(top_examples, top_sims):
        votes[ex["label"]] += sim

    total_weight = sum(votes.values()) or 1.0
    scores = {k: round(v / total_weight, 4) for k, v in votes.items()}
    label = max(votes, key=votes.get)
    confidence = scores[label]
    generation_ms = (time.perf_counter() - generation_start) * 1000

    total_ms = (time.perf_counter() - start) * 1000

    return RAGSentimentResult(
        label=label,
        confidence=round(confidence, 4),
        scores=scores,
        latency_ms=round(total_ms, 1),
        model_name="bert-base-uncased + RAG (110M, not fine-tuned)",
        retrieved_examples=[
            {**ex, "similarity": round(sim, 3)}
            for ex, sim in zip(top_examples, top_sims)
        ],
        retrieval_ms=round(retrieval_ms, 1),
        generation_ms=round(generation_ms, 1),
        is_live=True,
    )


# ---------------------------------------------------------------------------
# Non-streaming model calls (used by comparison pages)
# ---------------------------------------------------------------------------
def call_base_model(question: str, table: str = "", context: str = "") -> LLMResponse:
    client = _get_llm_client()
    if not client:
        return LLMResponse(answer=_llm_error_message(), latency_ms=0,
                           model_name="unavailable", is_live=False)
    prompt = _build_base_prompt(question, table, context)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=600,
    )
    ms = (time.perf_counter() - start) * 1000
    return LLMResponse(
        answer=resp.choices[0].message.content,
        latency_ms=round(ms, 1),
        model_name=f"{LLM_MODEL} (base)",
        is_live=True,
    )


def call_rag_model(question: str, table: str = "", context: str = "",
                   retrieved_docs: Optional[List[str]] = None) -> LLMResponse:
    client = _get_llm_client()
    if not client:
        return LLMResponse(answer=_llm_error_message(), latency_ms=0,
                           model_name="unavailable", is_live=False)
    if retrieved_docs is None:
        rag = _get_rag()
        result = rag.retrieve(question, top_k=3)
        retrieved_docs = result.documents
    prompt = _build_rag_prompt(question, table, context, retrieved_docs)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=800,
    )
    ms = (time.perf_counter() - start) * 1000
    return LLMResponse(
        answer=resp.choices[0].message.content,
        latency_ms=round(ms, 1),
        model_name=f"{LLM_MODEL} + RAG",
        retrieved_context=retrieved_docs,
        is_live=True,
    )


def call_finetuned_model(question: str, table: str = "", context: str = "") -> LLMResponse:
    """Call the actual FinQA-7B fine-tuned model (different weights from base).

    Uses FINETUNED_LLM_MODEL (finqa-7b) -- a Llama2-7B with weights updated
    by training on 8,281 FinQA financial reasoning examples.
    Same architecture as the base model, different weights.
    """
    client = _get_llm_client()
    if not client:
        return LLMResponse(answer=_llm_error_message(), latency_ms=0,
                           model_name="unavailable", is_live=False)
    prompt = _build_finetuned_prompt(question, table, context)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=FINETUNED_LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.1, max_tokens=800,
    )
    ms = (time.perf_counter() - start) * 1000
    return LLMResponse(
        answer=resp.choices[0].message.content,
        latency_ms=round(ms, 1),
        model_name=f"FinQA-7B (fine-tuned Llama2-7B)",
        is_live=True,
    )


def call_hybrid_model(question: str, table: str = "", context: str = "",
                      retrieved_docs: Optional[List[str]] = None) -> LLMResponse:
    """Hybrid: FinQA-7B (fine-tuned) + RAG retrieval.

    Uses the actual fine-tuned model with retrieved documents for context.
    Combines learned financial reasoning (weight changes) with fresh data (retrieval).
    """
    client = _get_llm_client()
    if not client:
        return LLMResponse(answer=_llm_error_message(), latency_ms=0,
                           model_name="unavailable", is_live=False)
    if retrieved_docs is None:
        rag = _get_rag()
        result = rag.retrieve(question, top_k=3)
        retrieved_docs = result.documents
    prompt = _build_hybrid_prompt(question, table, context, retrieved_docs)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=FINETUNED_LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.1, max_tokens=800,
    )
    ms = (time.perf_counter() - start) * 1000
    return LLMResponse(
        answer=resp.choices[0].message.content,
        latency_ms=round(ms, 1),
        model_name=f"FinQA-7B + RAG (fine-tuned Llama2-7B + retrieval)",
        retrieved_context=retrieved_docs,
        is_live=True,
    )


def retrieve_documents(query: str, top_k: int = 3):
    """Retrieve from real ChromaDB vector store."""
    rag = _get_rag()
    result = rag.retrieve(query, top_k=top_k)
    return result.documents, result.latency_ms


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
FINETUNED_SYSTEM_PROMPT = (
    "You are a financial analysis expert specializing in numerical reasoning "
    "over financial reports and SEC filings. When given financial tables and "
    "questions, you MUST:\n"
    "1. Identify the relevant numbers from the data\n"
    "2. Show every calculation step explicitly\n"
    "3. Compute exact results (not approximations)\n"
    "4. State the final answer clearly\n"
    "Always show your work. Be precise with decimal places."
)


def _build_base_prompt(question: str, table: str, context: str) -> str:
    parts = ["Answer the following financial question.\n"]
    if table:
        parts.append(f"Data:\n{table}\n")
    if context:
        parts.append(f"Context: {context}\n")
    parts.append(f"Question: {question}")
    return "\n".join(parts)


def _build_rag_prompt(question: str, table: str, context: str,
                      docs: List[str]) -> str:
    parts = ["Answer the following financial question using the provided data "
             "and the retrieved reference documents. Show your reasoning.\n"]
    if table:
        parts.append(f"Data:\n{table}\n")
    if context:
        parts.append(f"Context: {context}\n")
    parts.append("Retrieved Reference Documents:")
    for i, doc in enumerate(docs, 1):
        parts.append(f"[Doc {i}]: {doc}")
    parts.append(f"\nQuestion: {question}")
    return "\n".join(parts)


def _build_finetuned_prompt(question: str, table: str, context: str) -> str:
    parts = []
    if table:
        parts.append(f"Financial Data:\n{table}\n")
    if context:
        parts.append(f"Context: {context}\n")
    parts.append(f"Question: {question}")
    return "\n".join(parts)


def _build_hybrid_prompt(question: str, table: str, context: str,
                         docs: List[str]) -> str:
    parts = []
    if table:
        parts.append(f"Primary Data:\n{table}\n")
    if context:
        parts.append(f"Context: {context}\n")
    parts.append("Retrieved Reference Documents:")
    for i, doc in enumerate(docs, 1):
        parts.append(f"[Doc {i}]: {doc}")
    parts.append(f"\nQuestion: {question}\n\n"
                 "Provide step-by-step reasoning with precise calculations. "
                 "Reference the retrieved documents where relevant.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Streaming generators
# ---------------------------------------------------------------------------
def _stream_from_llm(messages: List[Dict], temperature: float = 0.3,
                     max_tokens: int = 800,
                     model: Optional[str] = None) -> Generator[str, None, None]:
    """Yield text chunks from Ollama via streaming API."""
    client = _get_llm_client()
    if not client:
        yield _llm_error_message()
        return
    try:
        stream = client.chat.completions.create(
            model=model or LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    except Exception as e:
        yield f"\n\n[Error: {e}]"


def stream_finetuned(question: str, table: str = "",
                     context: str = "") -> Generator[str, None, None]:
    """Stream from FinQA-7B (actual fine-tuned Llama2-7B)."""
    start = time.perf_counter()
    prompt = _build_finetuned_prompt(question, table, context)
    yield from _stream_from_llm([
        {"role": "user", "content": prompt},
    ], temperature=0.1, model=FINETUNED_LLM_MODEL)
    total_ms = (time.perf_counter() - start) * 1000
    yield (f"\n\n---\n"
           f"**Total time: {total_ms:.0f}ms** | No retrieval step\n\n"
           f"*Model: FinQA-7B (fine-tuned Llama2-7B, different weights)*")


def stream_rag(question: str, table: str = "",
               context: str = "") -> Generator[str, None, None]:
    """Stream RAG: retrieve real documents, then generate with Mistral.
    Shows output and timing for every step."""
    total_start = time.perf_counter()

    # Step 1: Embedding the query
    yield "**Step 1/3 -- Embedding query...**\n\n"
    embed_start = time.perf_counter()
    try:
        rag = _get_rag()
    except Exception as e:
        yield f"*RAG init error: {e}*\n\n"
        return
    embed_ms = (time.perf_counter() - embed_start) * 1000
    yield f"Query embedded into 384-dim vector ({embed_ms:.0f}ms)\n\n"

    # Step 2: Retrieval from ChromaDB
    yield "**Step 2/3 -- Searching vector store (ChromaDB)...**\n\n"
    retrieve_start = time.perf_counter()
    result = rag.retrieve(question, top_k=3)
    retrieve_ms = (time.perf_counter() - retrieve_start) * 1000

    for i, (doc, src, dist) in enumerate(
            zip(result.documents, result.sources, result.distances), 1):
        sim = 1 - dist
        yield (f"> **[Doc {i}]** _{src}_ (cosine similarity: {sim:.3f})\n"
               f"> {doc[:250]}...\n\n")
    yield (f"Retrieved **{len(result.documents)}** chunks from "
           f"{result.num_chunks_searched} indexed | "
           f"Retrieval time: **{retrieve_ms:.0f}ms**\n\n")
    docs = result.documents

    # Step 3: LLM Generation
    yield "**Step 3/3 -- LLM generating answer from retrieved context...**\n\n"
    gen_start = time.perf_counter()
    prompt = _build_rag_prompt(question, table, context, docs)
    yield from _stream_from_llm([{"role": "user", "content": prompt}])
    gen_ms = (time.perf_counter() - gen_start) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    yield (f"\n\n---\n"
           f"**Pipeline timing:**\n"
           f"| Step | Time |\n"
           f"|------|------|\n"
           f"| 1. Embed query | {embed_ms:.0f}ms |\n"
           f"| 2. Vector search | {retrieve_ms:.0f}ms |\n"
           f"| 3. LLM generation | {gen_ms:.0f}ms |\n"
           f"| **Total** | **{total_ms:.0f}ms** |\n\n"
           f"*Model: {LLM_MODEL} + RAG ({len(docs)} documents)*")


def stream_base(question: str, table: str = "",
                context: str = "") -> Generator[str, None, None]:
    """Stream from base Mistral -- no RAG, no system prompt."""
    start = time.perf_counter()
    prompt = _build_base_prompt(question, table, context)
    yield from _stream_from_llm([{"role": "user", "content": prompt}])
    total_ms = (time.perf_counter() - start) * 1000
    yield (f"\n\n---\n"
           f"**Total time: {total_ms:.0f}ms** | No retrieval, no system prompt\n\n"
           f"*Model: {LLM_MODEL} (base, no fine-tuning, no RAG)*")


def stream_hybrid(question: str, table: str = "",
                  context: str = "") -> Generator[str, None, None]:
    """Stream hybrid: RAG retrieval + fine-tuned-style generation.
    Shows per-step timing."""
    total_start = time.perf_counter()

    # Step 1: Retrieval
    yield "**Step 1/2 -- Retrieving supporting documents...**\n\n"
    retrieve_start = time.perf_counter()
    try:
        rag = _get_rag()
        result = rag.retrieve(question, top_k=3)
        retrieve_ms = (time.perf_counter() - retrieve_start) * 1000
        for i, (doc, src, dist) in enumerate(
                zip(result.documents, result.sources, result.distances), 1):
            sim = 1 - dist
            yield (f"> **[Doc {i}]** _{src}_ (cosine similarity: {sim:.3f})\n"
                   f"> {doc[:250]}...\n\n")
        yield (f"Retrieved **{len(result.documents)}** chunks | "
               f"Retrieval time: **{retrieve_ms:.0f}ms**\n\n")
        docs = result.documents
    except Exception as e:
        yield f"*Retrieval error: {e}*\n\n"
        docs = []
        retrieve_ms = 0

    # Step 2: Fine-tuned model generation with retrieved context
    yield "**Step 2/2 -- FinQA-7B generating with retrieved context...**\n\n"
    gen_start = time.perf_counter()
    prompt = _build_hybrid_prompt(question, table, context, docs)
    yield from _stream_from_llm([
        {"role": "user", "content": prompt},
    ], temperature=0.1, model=FINETUNED_LLM_MODEL)
    gen_ms = (time.perf_counter() - gen_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    yield (f"\n\n---\n"
           f"**Pipeline timing:**\n"
           f"| Step | Time |\n"
           f"|------|------|\n"
           f"| 1. Embed + retrieve | {retrieve_ms:.0f}ms |\n"
           f"| 2. FinQA-7B generation | {gen_ms:.0f}ms |\n"
           f"| **Total** | **{total_ms:.0f}ms** |\n\n"
           f"*Model: FinQA-7B + RAG (fine-tuned Llama2-7B + retrieval)*")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def get_demo_status() -> Dict[str, str]:
    status = {}
    status["finbert"] = "live" if finbert_available() else "unavailable"
    status["bert-base-rag"] = "live" if bert_base_available() else "unavailable"
    status["ollama"] = "live" if has_llm() else "unavailable"
    try:
        status["doc-rag"] = "live" if rag_ready() else "initializing"
    except Exception:
        status["doc-rag"] = "unavailable"
    return status
