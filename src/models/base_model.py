"""
Base Model (Mistral-7B) for RAG pipeline
Model: mistralai/Mistral-7B-Instruct-v0.2
"""
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ..config import model_config, get_device, is_gpu_available


@dataclass
class BaseModelResponse:
    """Response from base model (used in RAG)"""
    answer: str
    latency_ms: float
    tokens_generated: int
    context_used: str
    model_name: str = "Mistral-7B-Instruct"


class BaseModel:
    """
    General-purpose base model for RAG pipeline.
    Uses Mistral-7B-Instruct as the generation backbone.
    """

    def __init__(self, model_id: Optional[str] = None, load_in_4bit: bool = True):
        self.model_id = model_id or model_config.BASE_MODEL_ID
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.load_in_4bit = load_in_4bit and is_gpu_available()
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer"""
        if self._loaded:
            return

        print(f"Loading base model: {self.model_id}")
        print(f"Device: {self.device}, 4-bit quantization: {self.load_in_4bit}")

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto" if is_gpu_available() else None,
            torch_dtype=torch.float16 if is_gpu_available() else torch.float32,
        )

        if not is_gpu_available():
            self.model = self.model.to(self.device)

        self._loaded = True
        print("Base model loaded successfully!")

    def _format_rag_prompt(
        self,
        question: str,
        context_docs: List[str],
        table: Optional[str] = None
    ) -> str:
        """Format prompt for RAG-style generation"""
        context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])

        if table:
            prompt = f"""<s>[INST] You are a financial analyst assistant. Use the following retrieved documents and data to answer the question.

RETRIEVED CONTEXT:
{context}

TABLE DATA:
{table}

QUESTION: {question}

Please provide a clear, accurate answer based on the information provided. If you need to perform calculations, show your work step by step.

[/INST]"""
        else:
            prompt = f"""<s>[INST] You are a financial analyst assistant. Use the following retrieved documents to answer the question.

RETRIEVED CONTEXT:
{context}

QUESTION: {question}

Please provide a clear, accurate answer based on the information provided.

[/INST]"""

        return prompt

    def generate_with_context(
        self,
        question: str,
        context_docs: List[str],
        table: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1
    ) -> BaseModelResponse:
        """
        Generate an answer using retrieved context (RAG-style).

        Args:
            question: The question to answer
            context_docs: Retrieved documents for context
            table: Optional table data
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            BaseModelResponse with answer and metrics
        """
        if not self._loaded:
            self.load()

        # Format prompt
        prompt = self._format_rag_prompt(question, context_docs, table)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with timing
        start_time = time.perf_counter()

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

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Decode response
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up response
        answer = response_text.strip()

        return BaseModelResponse(
            answer=answer,
            latency_ms=latency_ms,
            tokens_generated=len(generated_tokens),
            context_used="\n---\n".join(context_docs[:2])  # First 2 docs for display
        )

    def generate_for_sentiment(
        self,
        text: str,
        context_docs: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.1
    ) -> BaseModelResponse:
        """
        Generate sentiment classification using RAG approach.

        Args:
            text: Text to classify
            context_docs: Retrieved sentiment examples/definitions

        Returns:
            BaseModelResponse with sentiment classification
        """
        if not self._loaded:
            self.load()

        prompt = f"""<s>[INST] You are a financial sentiment analyst. Based on the following examples and definitions, classify the sentiment of the given text.

REFERENCE INFORMATION:
{chr(10).join(context_docs)}

TEXT TO CLASSIFY:
"{text}"

Classify this text as either POSITIVE, NEGATIVE, or NEUTRAL. Provide only the classification and a brief explanation.

[/INST]"""

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,  # Deterministic for classification
                pad_token_id=self.tokenizer.pad_token_id
            )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Decode
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return BaseModelResponse(
            answer=response_text.strip(),
            latency_ms=latency_ms,
            tokens_generated=len(generated_tokens),
            context_used="\n---\n".join(context_docs[:2])
        )

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
_base_model: Optional[BaseModel] = None


def get_base_model() -> BaseModel:
    """Get or create the base model instance"""
    global _base_model
    if _base_model is None:
        _base_model = BaseModel()
    return _base_model
