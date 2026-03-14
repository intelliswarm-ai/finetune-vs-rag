"""
FinQA Dataset Loader
Dataset: ibm-research/finqa - 8,281 Q&A pairs for numerical reasoning
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from datasets import load_dataset

from ..config import dataset_config


@dataclass
class FinQAExample:
    """Single FinQA example"""
    id: str
    table: str  # Markdown formatted table
    text: str   # Pre-text and post-text combined
    question: str
    answer: str
    program: str  # The reasoning program/steps


def format_table_to_markdown(table_data: List[List[str]]) -> str:
    """Convert table data to markdown format"""
    if not table_data:
        return ""

    # Header row
    header = table_data[0]
    markdown = "| " + " | ".join(str(cell) for cell in header) + " |\n"
    markdown += "|" + "|".join(["---"] * len(header)) + "|\n"

    # Data rows
    for row in table_data[1:]:
        markdown += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    return markdown


def load_finqa_dataset(
    split: str = "test",
    num_samples: Optional[int] = None
) -> List[FinQAExample]:
    """
    Load FinQA dataset from HuggingFace.

    Args:
        split: Dataset split ('train', 'validation', 'test')
        num_samples: Number of samples to load (None for all)

    Returns:
        List of FinQAExample objects
    """
    print(f"Loading FinQA dataset (split={split})...")

    try:
        dataset = load_dataset(dataset_config.FINQA_DATASET_ID, split=split)
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Using sample data instead...")
        return get_sample_finqa_examples()

    examples = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Format table
        table_md = format_table_to_markdown(item.get("table", []))

        # Combine pre and post text
        pre_text = " ".join(item.get("pre_text", []))
        post_text = " ".join(item.get("post_text", []))
        full_text = f"{pre_text}\n\n{post_text}".strip()

        example = FinQAExample(
            id=item.get("id", f"finqa_{i}"),
            table=table_md,
            text=full_text,
            question=item.get("question", ""),
            answer=str(item.get("answer", "")),
            program=item.get("program", "")
        )
        examples.append(example)

    print(f"Loaded {len(examples)} FinQA examples")
    return examples


def get_sample_finqa_examples() -> List[FinQAExample]:
    """
    Return sample FinQA-style examples for demo purposes.
    These are representative examples that showcase numerical reasoning.
    """
    samples = [
        FinQAExample(
            id="demo_1",
            table="""| Segment | 2023 | 2022 | 2021 |
|---------|------|------|------|
| Consumer Banking | $12,450 | $11,200 | $10,500 |
| Commercial Banking | $8,320 | $7,890 | $7,200 |
| Investment Banking | $5,180 | $6,240 | $5,800 |""",
            text="The decrease in Investment Banking segment revenue was primarily driven by lower trading volumes and reduced M&A advisory fees due to market uncertainty. Consumer Banking continued to show strong growth driven by higher net interest income.",
            question="What was the percentage change in total revenue from 2022 to 2023?",
            answer="2.45%",
            program="add(11200, 7890, 6240) = 25330; add(12450, 8320, 5180) = 25950; subtract(25950, 25330) = 620; divide(620, 25330) = 0.0245"
        ),
        FinQAExample(
            id="demo_2",
            table="""| Item | 2023 | 2022 |
|------|------|------|
| Total Assets | $245,600 | $231,400 |
| Total Liabilities | $198,200 | $187,600 |
| Shareholders' Equity | $47,400 | $43,800 |
| Total Debt | $52,300 | $48,900 |""",
            text="The company maintained a strong capital position throughout 2023, with equity growing faster than debt. Management continues to focus on deleveraging the balance sheet.",
            question="Calculate the debt-to-equity ratio for 2023 and compare it to 2022. Is leverage increasing or decreasing?",
            answer="2023 D/E: 1.10, 2022 D/E: 1.12. Leverage is decreasing.",
            program="divide(52300, 47400) = 1.10; divide(48900, 43800) = 1.12"
        ),
        FinQAExample(
            id="demo_3",
            table="""| Metric | Q4 2023 | Q3 2023 | Q4 2022 |
|--------|---------|---------|---------|
| Net Interest Income | $14,200 | $13,800 | $12,500 |
| Non-Interest Income | $5,600 | $5,900 | $6,100 |
| Total Revenue | $19,800 | $19,700 | $18,600 |
| Operating Expenses | $11,200 | $10,900 | $10,400 |""",
            text="Net interest income continued to benefit from higher rates, while non-interest income faced headwinds from reduced trading activity. The efficiency ratio improved year-over-year.",
            question="What is the efficiency ratio (Operating Expenses / Total Revenue) for Q4 2023, and how does it compare to Q4 2022?",
            answer="Q4 2023: 56.6%, Q4 2022: 55.9%. Efficiency ratio increased by 0.7 percentage points.",
            program="divide(11200, 19800) = 0.566; divide(10400, 18600) = 0.559"
        ),
        FinQAExample(
            id="demo_4",
            table="""| Region | Revenue 2023 | Revenue 2022 | Employees |
|--------|--------------|--------------|-----------|
| North America | $45,200 | $42,100 | 12,500 |
| Europe | $28,700 | $31,200 | 8,200 |
| Asia Pacific | $18,900 | $16,400 | 5,800 |""",
            text="North America showed robust growth driven by strong consumer demand. Europe faced currency headwinds and economic uncertainty. Asia Pacific emerged as the fastest-growing region.",
            question="What was the growth rate for Asia Pacific, and what is the revenue per employee in that region for 2023?",
            answer="Asia Pacific growth: 15.2%, Revenue per employee: $3.26M",
            program="subtract(18900, 16400) = 2500; divide(2500, 16400) = 0.152; divide(18900, 5800) = 3.26"
        ),
        FinQAExample(
            id="demo_5",
            table="""| Quarter | EPS | Dividend | Payout Ratio |
|---------|-----|----------|--------------|
| Q1 2023 | $2.15 | $0.55 | 25.6% |
| Q2 2023 | $2.28 | $0.55 | 24.1% |
| Q3 2023 | $2.42 | $0.60 | 24.8% |
| Q4 2023 | $2.55 | $0.60 | 23.5% |""",
            text="The company increased its quarterly dividend by 9% in Q3 2023, reflecting confidence in sustained earnings growth. Full year EPS of $9.40 exceeded guidance.",
            question="What was the total dividend paid for the full year 2023, and what percentage of annual EPS does this represent?",
            answer="Total dividend: $2.30, Payout ratio: 24.5%",
            program="add(0.55, 0.55, 0.60, 0.60) = 2.30; add(2.15, 2.28, 2.42, 2.55) = 9.40; divide(2.30, 9.40) = 0.245"
        )
    ]

    return samples


def get_demo_examples(n: int = 5) -> List[FinQAExample]:
    """Get n sample examples for demo purposes"""
    samples = get_sample_finqa_examples()
    return samples[:n]
