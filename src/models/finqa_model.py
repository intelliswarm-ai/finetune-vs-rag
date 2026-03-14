"""
FinQA-7B Model for numerical reasoning over financial reports
Model: truocpham/FinQA-7B-Instruct-v0.1
"""
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ..config import model_config, get_device, is_gpu_available


@dataclass
class FinQAResponse:
    """Response from FinQA model"""
    answer: str
    reasoning_steps: str
    latency_ms: float
    tokens_generated: int
    model_name: str = "FinQA-7B-Instruct"


class FinQAModel:
    """
    Fine-tuned model for numerical reasoning over financial data.
    Specialized for FinQA benchmark tasks: calculations, ratios, percentages.
    """

    def __init__(self, model_id: Optional[str] = None, load_in_4bit: bool = True):
        self.model_id = model_id or model_config.FINQA_MODEL_ID
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.load_in_4bit = load_in_4bit and is_gpu_available()
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer"""
        if self._loaded:
            return

        print(f"Loading FinQA model: {self.model_id}")
        print(f"Device: {self.device}, 4-bit quantization: {self.load_in_4bit}")

        # Configure quantization for memory efficiency
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
        print("FinQA model loaded successfully!")

    def _format_prompt(self, table: str, text: str, question: str) -> str:
        """Format the input for FinQA-style numerical reasoning"""
        prompt = f"""You are a financial analyst expert at numerical reasoning.

Given the following financial data:

TABLE:
{table}

CONTEXT:
{text}

QUESTION: {question}

Please solve this step by step:
1. Identify the relevant numbers from the table and context
2. Determine the calculation needed
3. Perform the calculation
4. Provide the final answer

SOLUTION:"""
        return prompt

    def generate(
        self,
        table: str,
        text: str,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1
    ) -> FinQAResponse:
        """
        Generate an answer for a financial numerical reasoning question.

        Args:
            table: Financial table data (markdown format)
            text: Accompanying text context
            question: The question to answer
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            FinQAResponse with answer, reasoning, and metrics
        """
        if not self._loaded:
            self.load()

        # Format prompt
        prompt = self._format_prompt(table, text, question)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
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

        # Parse reasoning and answer
        reasoning_steps, answer = self._parse_response(response_text)

        return FinQAResponse(
            answer=answer,
            reasoning_steps=reasoning_steps,
            latency_ms=latency_ms,
            tokens_generated=len(generated_tokens)
        )

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the model response into reasoning steps and final answer"""
        lines = response.strip().split("\n")

        # Find the final answer
        answer = ""
        reasoning_lines = []

        for line in lines:
            if line.lower().startswith("answer:") or line.lower().startswith("final answer:"):
                answer = line.split(":", 1)[-1].strip()
            else:
                reasoning_lines.append(line)

        # If no explicit answer marker, use the last line
        if not answer and lines:
            answer = lines[-1].strip()

        reasoning_steps = "\n".join(reasoning_lines).strip()

        return reasoning_steps, answer

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


# Singleton instance for reuse
_finqa_model: Optional[FinQAModel] = None


def get_finqa_model() -> FinQAModel:
    """Get or create the FinQA model instance"""
    global _finqa_model
    if _finqa_model is None:
        _finqa_model = FinQAModel()
    return _finqa_model
