"""
Hybrid Model: Fine-tuned FinQA-7B enhanced with RAG retrieval
Best of both worlds: Domain expertise + Fresh context
"""
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ..config import model_config, get_device, is_gpu_available
from ..rag.vector_store import VectorStore, get_vector_store
from ..rag.embeddings import get_embedding_model


@dataclass
class HybridResponse:
    """Response from hybrid model (Fine-tuned + RAG)"""
    answer: str
    reasoning_steps: str
    retrieved_docs: List[str]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    tokens_generated: int
    model_name: str = "Hybrid (FinQA-7B + RAG)"


class HybridModel:
    """
    Hybrid approach: Use FinQA fine-tuned model with RAG-retrieved context.

    This combines:
    - FinQA-7B's specialized numerical reasoning capabilities
    - RAG's ability to incorporate relevant external context

    Best for: Complex financial questions that benefit from both
    domain expertise AND specific document context.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        vector_store: VectorStore = None,
        load_in_4bit: bool = True
    ):
        self.model_id = model_id or model_config.FINQA_MODEL_ID
        self.device = get_device()
        self.vector_store = vector_store or get_vector_store()
        self.model = None
        self.tokenizer = None
        self.load_in_4bit = load_in_4bit and is_gpu_available()
        self._loaded = False

    def load(self) -> None:
        """Load the fine-tuned model and initialize vector store"""
        if self._loaded:
            return

        print(f"Loading Hybrid model: {self.model_id}")
        print(f"Device: {self.device}, 4-bit quantization: {self.load_in_4bit}")

        # Initialize vector store
        self.vector_store.initialize()

        # Configure quantization
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto" if is_gpu_available() else None,
            torch_dtype=torch.float16 if is_gpu_available() else torch.float32,
            trust_remote_code=True
        )

        if not is_gpu_available():
            self.model = self.model.to(self.device)

        self._loaded = True
        print("Hybrid model loaded successfully!")

    def _format_hybrid_prompt(
        self,
        table: str,
        text: str,
        question: str,
        retrieved_docs: List[str]
    ) -> str:
        """Format prompt with both original data and retrieved context"""

        # Format retrieved context
        context_section = ""
        if retrieved_docs:
            context_section = "\nADDITIONAL CONTEXT FROM KNOWLEDGE BASE:\n"
            for i, doc in enumerate(retrieved_docs[:3], 1):  # Top 3 docs
                context_section += f"\n[Reference {i}]: {doc[:500]}...\n" if len(doc) > 500 else f"\n[Reference {i}]: {doc}\n"

        prompt = f"""You are a financial analyst expert at numerical reasoning.
You have access to both the primary financial data AND additional context from a knowledge base.

PRIMARY DATA:

TABLE:
{table}

CONTEXT:
{text}
{context_section}
QUESTION: {question}

Please solve this step by step using both the primary data and any relevant additional context:
1. Identify the relevant numbers from the table and context
2. Consider any additional information from the knowledge base
3. Determine the calculation needed
4. Perform the calculation
5. Provide the final answer

SOLUTION:"""
        return prompt

    def generate(
        self,
        table: str,
        text: str,
        question: str,
        n_retrieve: int = 3,
        max_new_tokens: int = 512,
        temperature: float = 0.1
    ) -> HybridResponse:
        """
        Generate answer using fine-tuned model with RAG-enhanced context.

        Args:
            table: Financial table data
            text: Accompanying text context
            question: The question to answer
            n_retrieve: Number of documents to retrieve
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            HybridResponse with answer, reasoning, retrieved docs, and metrics
        """
        if not self._loaded:
            self.load()

        total_start = time.perf_counter()

        # Step 1: Retrieve relevant documents
        retrieval_start = time.perf_counter()
        retrieval_result = self.vector_store.query(question, n_results=n_retrieve)
        retrieved_docs = retrieval_result.documents
        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000

        # Step 2: Format prompt with retrieved context
        prompt = self._format_hybrid_prompt(table, text, question, retrieved_docs)

        # Step 3: Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072  # Larger context for hybrid
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Step 4: Generate with timing
        gen_start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=model_config.TOP_P,
                do_sample=model_config.DO_SAMPLE,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        gen_latency = (time.perf_counter() - gen_start) * 1000
        total_latency = (time.perf_counter() - total_start) * 1000

        # Decode response
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Parse reasoning and answer
        reasoning_steps, answer = self._parse_response(response_text)

        return HybridResponse(
            answer=answer,
            reasoning_steps=reasoning_steps,
            retrieved_docs=retrieved_docs,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=gen_latency,
            total_latency_ms=total_latency,
            tokens_generated=len(generated_tokens)
        )

    def _parse_response(self, response: str) -> tuple:
        """Parse the model response into reasoning steps and final answer"""
        lines = response.strip().split("\n")

        answer = ""
        reasoning_lines = []

        for line in lines:
            if line.lower().startswith("answer:") or line.lower().startswith("final answer:"):
                answer = line.split(":", 1)[-1].strip()
            else:
                reasoning_lines.append(line)

        if not answer and lines:
            answer = lines[-1].strip()

        reasoning_steps = "\n".join(reasoning_lines).strip()

        return reasoning_steps, answer

    def add_to_knowledge_base(self, documents: List[str], metadatas: List[Dict] = None) -> int:
        """Add documents to the RAG knowledge base"""
        return self.vector_store.add_documents(documents, metadatas)

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded

    def unload(self) -> None:
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        if is_gpu_available():
            torch.cuda.empty_cache()


# Singleton instance
_hybrid_model: Optional[HybridModel] = None


def get_hybrid_model() -> HybridModel:
    """Get or create the hybrid model instance"""
    global _hybrid_model
    if _hybrid_model is None:
        _hybrid_model = HybridModel()
    return _hybrid_model
