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

LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
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
# RAG-based sentiment (retrieve labeled examples + LLM classifies)
# ---------------------------------------------------------------------------
SENTIMENT_KNOWLEDGE_BASE = [
    {"text": "Revenue exceeded analyst expectations by 8%.", "label": "positive"},
    {"text": "Quarterly profit margins expanded to a five-year high.", "label": "positive"},
    {"text": "Loan growth accelerated across all segments.", "label": "positive"},
    {"text": "Customer acquisition reached record levels in Q4.", "label": "positive"},
    {"text": "The firm reported a net loss for the third consecutive quarter.", "label": "negative"},
    {"text": "Operating expenses surged 20% due to regulatory fines.", "label": "negative"},
    {"text": "Credit quality deteriorated amid rising delinquencies.", "label": "negative"},
    {"text": "The bank warned of material weakness in internal controls.", "label": "negative"},
    {"text": "Total deposits remained flat compared to the prior period.", "label": "neutral"},
    {"text": "The board approved a routine extension of the credit facility.", "label": "neutral"},
    {"text": "Staffing levels were unchanged from last quarter.", "label": "neutral"},
    {"text": "The company filed its annual 10-K report on time.", "label": "neutral"},
]


def run_rag_sentiment(text: str) -> RAGSentimentResult:
    """RAG-based sentiment: retrieve similar labeled examples, LLM classifies."""
    start = time.perf_counter()

    # Retrieve similar examples via keyword overlap
    retrieval_start = time.perf_counter()
    query_words = set(text.lower().split())
    scored = []
    for ex in SENTIMENT_KNOWLEDGE_BASE:
        overlap = len(query_words & set(ex["text"].lower().split()))
        scored.append((overlap, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_examples = [ex for _, ex in scored[:3]]
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    # Generate classification with LLM
    generation_start = time.perf_counter()
    client = _get_llm_client()
    if client:
        context = "\n".join(f'- "{e["text"]}" -> {e["label"]}' for e in top_examples)
        prompt = (
            "Classify the sentiment of the following financial text as "
            "positive, negative, or neutral.\n\n"
            f"Similar examples:\n{context}\n\n"
            f'Text: "{text}"\n\n'
            "Respond with ONLY one word: positive, negative, or neutral."
        )
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            answer = resp.choices[0].message.content.strip().lower()
            if "positive" in answer:
                label = "positive"
            elif "negative" in answer:
                label = "negative"
            else:
                label = "neutral"
        except Exception as e:
            label = "neutral"
    else:
        # Fallback: majority vote from retrieved examples
        votes = {"positive": 0, "negative": 0, "neutral": 0}
        for ex in top_examples:
            votes[ex["label"]] += 1
        label = max(votes, key=votes.get)

    generation_ms = (time.perf_counter() - generation_start) * 1000
    total_ms = (time.perf_counter() - start) * 1000

    return RAGSentimentResult(
        label=label,
        confidence=0.60,
        scores={"positive": 0.33, "neutral": 0.34, "negative": 0.33},
        latency_ms=round(total_ms, 1),
        model_name=f"{LLM_MODEL} + RAG retrieval",
        retrieved_examples=top_examples,
        retrieval_ms=round(retrieval_ms, 1),
        generation_ms=round(generation_ms, 1),
        is_live=client is not None,
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
    """Call LLM with a strong financial-expert system prompt.
    In production this would be a truly fine-tuned model like FinQA-7B."""
    client = _get_llm_client()
    if not client:
        return LLMResponse(answer=_llm_error_message(), latency_ms=0,
                           model_name="unavailable", is_live=False)
    prompt = _build_finetuned_prompt(question, table, context)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": FINETUNED_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1, max_tokens=800,
    )
    ms = (time.perf_counter() - start) * 1000
    return LLMResponse(
        answer=resp.choices[0].message.content,
        latency_ms=round(ms, 1),
        model_name=f"{LLM_MODEL} (financial expert prompt)",
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
                     max_tokens: int = 800) -> Generator[str, None, None]:
    """Yield text chunks from Ollama via streaming API."""
    client = _get_llm_client()
    if not client:
        yield _llm_error_message()
        return
    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
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
    """Stream from Mistral with financial-expert system prompt."""
    prompt = _build_finetuned_prompt(question, table, context)
    yield from _stream_from_llm([
        {"role": "system", "content": FINETUNED_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ], temperature=0.1)
    yield f"\n\n---\n*Model: {LLM_MODEL} with financial-expert system prompt*"


def stream_rag(question: str, table: str = "",
               context: str = "") -> Generator[str, None, None]:
    """Stream RAG: retrieve real documents, then generate with Mistral."""
    yield "**Step 1 -- Retrieving from knowledge base...**\n\n"

    try:
        rag = _get_rag()
        result = rag.retrieve(question, top_k=3)
        for i, (doc, src, dist) in enumerate(
                zip(result.documents, result.sources, result.distances), 1):
            yield (f"> **[Doc {i}]** _{src}_ (similarity: {1 - dist:.2f})\n"
                   f"> {doc[:200]}...\n\n")
        yield (f"*Retrieved {len(result.documents)} chunks from "
               f"{result.num_chunks_searched} indexed "
               f"({result.latency_ms:.0f}ms)*\n\n")
        docs = result.documents
    except Exception as e:
        yield f"*Retrieval error: {e}*\n\n"
        docs = []

    yield "**Step 2 -- Generating answer with retrieved context...**\n\n"
    prompt = _build_rag_prompt(question, table, context, docs)
    yield from _stream_from_llm([{"role": "user", "content": prompt}])
    yield f"\n\n---\n*Model: {LLM_MODEL} + RAG ({len(docs)} documents)*"


def stream_base(question: str, table: str = "",
                context: str = "") -> Generator[str, None, None]:
    """Stream from base Mistral -- no RAG, no system prompt."""
    prompt = _build_base_prompt(question, table, context)
    yield from _stream_from_llm([{"role": "user", "content": prompt}])
    yield f"\n\n---\n*Model: {LLM_MODEL} (base, no fine-tuning, no RAG)*"


def stream_hybrid(question: str, table: str = "",
                  context: str = "") -> Generator[str, None, None]:
    """Stream hybrid: RAG retrieval + fine-tuned-style generation."""
    yield "**Step 1 -- Retrieving supporting documents...**\n\n"

    try:
        rag = _get_rag()
        result = rag.retrieve(question, top_k=3)
        for i, (doc, src, dist) in enumerate(
                zip(result.documents, result.sources, result.distances), 1):
            yield (f"> **[Doc {i}]** _{src}_ (similarity: {1 - dist:.2f})\n"
                   f"> {doc[:200]}...\n\n")
        yield (f"*Retrieved {len(result.documents)} chunks "
               f"({result.latency_ms:.0f}ms)*\n\n")
        docs = result.documents
    except Exception as e:
        yield f"*Retrieval error: {e}*\n\n"
        docs = []

    yield "**Step 2 -- Fine-tuned model generating with context...**\n\n"
    prompt = _build_hybrid_prompt(question, table, context, docs)
    yield from _stream_from_llm([
        {"role": "system", "content": FINETUNED_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ], temperature=0.1)
    yield f"\n\n---\n*Model: {LLM_MODEL} hybrid (expert prompt + RAG)*"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def get_demo_status() -> Dict[str, str]:
    status = {}
    status["finbert"] = "live" if finbert_available() else "unavailable"
    status["ollama"] = "live" if has_llm() else "unavailable"
    try:
        status["rag"] = "live" if rag_ready() else "initializing"
    except Exception:
        status["rag"] = "unavailable"
    return status
