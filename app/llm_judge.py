"""
LLM-as-Judge Evaluation Module

Uses OpenAI GPT-4o (or configurable model) to evaluate model outputs
with structured scoring:
  - Correctness (1-5): Is the answer factually correct?
  - Reasoning Quality (1-5): Does the answer show sound reasoning steps?
  - Faithfulness (1-5): Does the answer stay faithful to the provided data?

References:
  - Paper #5: Pints AI Bench-RAG with GPT-4o judge
  - Paper #2: INLG -- auto metrics can mislead

Configuration via environment variables:
  - OPENAI_API_KEY: Required. Your OpenAI API key.
  - JUDGE_MODEL: Optional. Defaults to "gpt-4o". Other options: "gpt-4o-mini", "gpt-4-turbo", etc.
"""
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")


@dataclass
class JudgeScore:
    correctness: int          # 1-5
    reasoning_quality: int    # 1-5
    faithfulness: int         # 1-5
    overall: float            # weighted average
    explanation: str
    judge_model: str
    latency_ms: float


def _get_judge_client():
    """Return an OpenAI client using the OpenAI API."""
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client
    except Exception:
        return None


def judge_available() -> bool:
    """Check if the OpenAI judge is available (API key is set)."""
    return _get_judge_client() is not None


def get_judge_model_name() -> Optional[str]:
    """Return the name of the judge model that would be used."""
    if not OPENAI_API_KEY:
        return None
    return JUDGE_MODEL


# -------------------------------------------------------------------------
# Prompt templates per task type
# -------------------------------------------------------------------------

_SENTIMENT_JUDGE_PROMPT = """You are an expert evaluator assessing the quality of a financial sentiment classification.

TASK: Evaluate how well the model classified the sentiment of a financial text.

INPUT TEXT: {input_text}
EXPECTED LABEL: {expected}
MODEL'S PREDICTION: {model_output}
MODEL'S CONFIDENCE: {confidence}

Score each dimension from 1 (worst) to 5 (best):

1. CORRECTNESS (1-5): Does the predicted label match the expected label?
   - 5: Correct label
   - 3: Related but wrong (e.g., neutral instead of negative for a mildly negative text)
   - 1: Completely wrong (e.g., positive for clearly negative text)

2. REASONING QUALITY (1-5): Based on the text and prediction, how well does the model appear to understand the financial language?
   - 5: Prediction shows deep understanding of financial jargon and nuance
   - 3: Prediction shows basic understanding
   - 1: Prediction appears random or misunderstands financial terms

3. FAITHFULNESS (1-5): Does the prediction stay faithful to what the text actually says?
   - 5: Prediction is fully grounded in the text content
   - 3: Partially grounded
   - 1: Prediction seems to hallucinate or ignore text content

Respond ONLY with valid JSON in this exact format:
{{"correctness": <1-5>, "reasoning_quality": <1-5>, "faithfulness": <1-5>, "explanation": "<one sentence>"}}"""


_NUMERICAL_JUDGE_PROMPT = """You are an expert evaluator assessing the quality of a financial numerical reasoning answer.

TASK: Evaluate how well the model answered a financial calculation question.

QUESTION: {question}
DATA TABLE:
{table}
CONTEXT: {context}
EXPECTED ANSWER: {expected}
MODEL'S ANSWER: {model_output}

Score each dimension from 1 (worst) to 5 (best):

1. CORRECTNESS (1-5): How close is the numerical answer to the expected value?
   - 5: Exact match or within 2% of expected
   - 4: Within 10% of expected
   - 3: Within 25% of expected
   - 2: Within 50% of expected
   - 1: More than 50% off or no numerical answer given

2. REASONING QUALITY (1-5): Does the answer show correct mathematical steps?
   - 5: Complete, step-by-step reasoning with correct intermediate values
   - 4: Shows key steps, minor omissions
   - 3: Shows some reasoning but with errors in logic
   - 2: Minimal reasoning, mostly guessing
   - 1: No reasoning shown or completely wrong approach

3. FAITHFULNESS (1-5): Does the answer use ONLY the data provided in the table and context?
   - 5: All numbers come from the provided data
   - 3: Mostly uses provided data but introduces some external assumptions
   - 1: Uses fabricated numbers or ignores provided data

Respond ONLY with valid JSON in this exact format:
{{"correctness": <1-5>, "reasoning_quality": <1-5>, "faithfulness": <1-5>, "explanation": "<one sentence>"}}"""


_SPAM_JUDGE_PROMPT = """You are an expert evaluator assessing the quality of a spam/phishing email classification.

TASK: Evaluate how well the model classified an email as spam or legitimate (ham).

EMAIL TEXT: {input_text}
EXPECTED LABEL: {expected}
MODEL'S PREDICTION: {model_output}
MODEL'S CONFIDENCE: {confidence}

Score each dimension from 1 (worst) to 5 (best):

1. CORRECTNESS (1-5): Does the predicted label match the expected label?
   - 5: Correct label
   - 3: Wrong but understandable (e.g., legitimate email with spam-like language)
   - 1: Completely wrong for an obvious case

2. REASONING QUALITY (1-5): How well does the model appear to distinguish spam/phishing signals from legitimate patterns?
   - 5: Correctly identifies phishing cues (urgency, fake links, personal info requests) or legitimate signals (known sender patterns, internal references)
   - 3: Basic pattern matching without nuance
   - 1: No apparent understanding of spam vs ham signals

3. FAITHFULNESS (1-5): Does the classification appear to be based on the actual email content?
   - 5: Classification clearly follows from email content analysis
   - 3: Classification is loosely related to content
   - 1: Classification seems random or based on superficial keyword matching

Respond ONLY with valid JSON in this exact format:
{{"correctness": <1-5>, "reasoning_quality": <1-5>, "faithfulness": <1-5>, "explanation": "<one sentence>"}}"""


def _parse_judge_response(text: str) -> Optional[Dict]:
    """Extract JSON scores from judge model response."""
    # Try direct JSON parse
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        data = json.loads(text)
        if all(k in data for k in ("correctness", "reasoning_quality", "faithfulness")):
            # Clamp values to 1-5
            for k in ("correctness", "reasoning_quality", "faithfulness"):
                data[k] = max(1, min(5, int(data[k])))
            return data
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: regex extraction
    scores = {}
    for key in ("correctness", "reasoning_quality", "faithfulness"):
        m = re.search(rf'"{key}"\s*:\s*(\d)', text)
        if m:
            scores[key] = max(1, min(5, int(m.group(1))))
    if len(scores) == 3:
        # Try to get explanation
        exp_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', text)
        scores["explanation"] = exp_match.group(1) if exp_match else ""
        return scores

    return None


def judge_sentiment(input_text: str, expected: str, model_output: str,
                    confidence: float = 0.0, judge_model: str = None) -> Optional[JudgeScore]:
    """Judge a sentiment classification result."""
    prompt = _SENTIMENT_JUDGE_PROMPT.format(
        input_text=input_text,
        expected=expected,
        model_output=model_output,
        confidence=f"{confidence:.3f}" if confidence else "N/A",
    )
    return _call_judge(prompt, judge_model)


def judge_numerical(question: str, table: str, context: str, expected: str,
                    model_output: str, judge_model: str = None) -> Optional[JudgeScore]:
    """Judge a numerical reasoning result."""
    prompt = _NUMERICAL_JUDGE_PROMPT.format(
        question=question,
        table=table,
        context=context,
        expected=expected,
        model_output=model_output,
    )
    return _call_judge(prompt, judge_model)


def judge_spam(input_text: str, expected: str, model_output: str,
               confidence: float = 0.0, judge_model: str = None) -> Optional[JudgeScore]:
    """Judge a spam classification result."""
    prompt = _SPAM_JUDGE_PROMPT.format(
        input_text=input_text,
        expected=expected,
        model_output=model_output,
        confidence=f"{confidence:.3f}" if confidence else "N/A",
    )
    return _call_judge(prompt, judge_model)


_RETRIEVAL_QA_JUDGE_PROMPT = """You are an expert evaluator assessing the quality of a retrieval-augmented question answering response about a financial institution (Meridian National Bancorp).

TASK: Evaluate how well the model answered a factual question that requires retrieving information from financial documents.

QUESTION: {question}
EXPECTED ANSWER: {expected}
MODEL'S ANSWER: {model_output}
SOURCE DOCUMENTS: {source_documents}
CATEGORY: {category}

Score each dimension from 1 (worst) to 5 (best):

1. CORRECTNESS (1-5): Does the answer contain the correct factual information?
   - 5: Contains the exact expected numbers/facts
   - 4: Contains the key facts with minor imprecision
   - 3: Partially correct -- some facts right, some wrong or missing
   - 2: Mostly incorrect but shows some relevant knowledge
   - 1: Completely wrong, refuses to answer, or hallucinates facts

2. REASONING QUALITY (1-5): Does the answer demonstrate understanding and provide useful context?
   - 5: Clear explanation with relevant context and interpretation
   - 4: Good explanation with minor gaps
   - 3: Basic answer without much context or explanation
   - 2: Vague or confusing response
   - 1: No meaningful reasoning

3. FAITHFULNESS (1-5): Is the answer grounded in the source documents (not hallucinated)?
   - 5: All facts clearly come from the documents, no fabrication
   - 4: Mostly grounded, minor inferences that are reasonable
   - 3: Mix of grounded facts and unverifiable claims
   - 2: Significant hallucination or fabricated details
   - 1: Answer is entirely fabricated or contradicts source documents

Respond ONLY with valid JSON in this exact format:
{{"correctness": <1-5>, "reasoning_quality": <1-5>, "faithfulness": <1-5>, "explanation": "<one sentence>"}}"""


def judge_retrieval_qa(question: str, expected: str, model_output: str,
                       source_documents: str = "", category: str = "",
                       judge_model: str = None) -> Optional[JudgeScore]:
    """Judge a retrieval-augmented QA result."""
    prompt = _RETRIEVAL_QA_JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        model_output=model_output[:800],
        source_documents=source_documents or "N/A",
        category=category or "general",
    )
    return _call_judge(prompt, judge_model)


def _call_judge(prompt: str, judge_model: str = None) -> Optional[JudgeScore]:
    """Call the judge model and parse response into a JudgeScore."""
    client = _get_judge_client()
    if client is None:
        return None

    model = judge_model or JUDGE_MODEL

    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise evaluation judge. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content
    except Exception:
        return None
    elapsed_ms = (time.perf_counter() - start) * 1000

    parsed = _parse_judge_response(raw)
    if parsed is None:
        return None

    overall = (
        parsed["correctness"] * 0.5 +
        parsed["reasoning_quality"] * 0.3 +
        parsed["faithfulness"] * 0.2
    )

    return JudgeScore(
        correctness=parsed["correctness"],
        reasoning_quality=parsed["reasoning_quality"],
        faithfulness=parsed["faithfulness"],
        overall=round(overall, 2),
        explanation=parsed.get("explanation", ""),
        judge_model=model,
        latency_ms=round(elapsed_ms, 1),
    )


def judge_score_to_dict(score: JudgeScore) -> Dict:
    """Convert JudgeScore to a JSON-serializable dict."""
    return asdict(score)


def compute_judge_summary(results: List[Dict], model_keys: List[str]) -> Dict:
    """Compute average judge scores across all results for each model."""
    summary = {}
    for m in model_keys:
        scores = {
            "correctness": [],
            "reasoning_quality": [],
            "faithfulness": [],
            "overall": [],
        }
        for r in results:
            judge = r.get(f"{m}_judge")
            if judge and isinstance(judge, dict):
                for k in scores:
                    v = judge.get(k)
                    if v is not None:
                        scores[k].append(float(v))

        if scores["overall"]:
            summary[m] = {
                k: round(sum(v) / len(v), 2) if v else 0
                for k, v in scores.items()
            }
            summary[m]["count"] = len(scores["overall"])
        else:
            summary[m] = {
                "correctness": 0, "reasoning_quality": 0,
                "faithfulness": 0, "overall": 0, "count": 0,
            }
    return summary
