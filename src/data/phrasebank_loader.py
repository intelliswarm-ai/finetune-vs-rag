"""
Financial PhraseBank Dataset Loader
Dataset: takala/financial_phrasebank - 5,000 labeled financial sentences
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass

from datasets import load_dataset

from ..config import dataset_config


@dataclass
class SentimentExample:
    """Single sentiment example"""
    text: str
    label: str  # positive, negative, neutral
    confidence: float  # Annotator agreement level


def load_phrasebank_dataset(
    agreement_level: str = "sentences_75agree",
    num_samples: Optional[int] = None
) -> List[SentimentExample]:
    """
    Load Financial PhraseBank dataset from HuggingFace.

    Args:
        agreement_level: Level of annotator agreement
            - 'sentences_50agree': 50% agreement
            - 'sentences_66agree': 66% agreement
            - 'sentences_75agree': 75% agreement (recommended)
            - 'sentences_allagree': 100% agreement
        num_samples: Number of samples to load (None for all)

    Returns:
        List of SentimentExample objects
    """
    print(f"Loading Financial PhraseBank (agreement={agreement_level})...")

    try:
        dataset = load_dataset(
            dataset_config.PHRASEBANK_DATASET_ID,
            agreement_level,
            split="train"
        )
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Using sample data instead...")
        return get_sample_sentiment_examples()

    # Label mapping
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    # Agreement confidence mapping
    confidence_map = {
        "sentences_50agree": 0.50,
        "sentences_66agree": 0.66,
        "sentences_75agree": 0.75,
        "sentences_allagree": 1.00
    }

    examples = []
    confidence = confidence_map.get(agreement_level, 0.75)

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        example = SentimentExample(
            text=item["sentence"],
            label=label_map.get(item["label"], "neutral"),
            confidence=confidence
        )
        examples.append(example)

    print(f"Loaded {len(examples)} sentiment examples")
    return examples


def get_sample_sentiment_examples() -> List[SentimentExample]:
    """
    Return sample sentiment examples for demo purposes.
    These are representative financial sentiment examples.
    """
    samples = [
        # Positive examples
        SentimentExample(
            text="Net interest income grew 12% driven by higher rates and loan growth.",
            label="positive",
            confidence=0.95
        ),
        SentimentExample(
            text="The company reported record quarterly revenue, exceeding analyst expectations.",
            label="positive",
            confidence=0.92
        ),
        SentimentExample(
            text="Operating margins expanded 150 basis points year-over-year.",
            label="positive",
            confidence=0.88
        ),
        SentimentExample(
            text="Strong demand in core markets contributed to double-digit growth.",
            label="positive",
            confidence=0.90
        ),

        # Negative examples
        SentimentExample(
            text="Management expects headwinds from deposit competition to persist.",
            label="negative",
            confidence=0.87
        ),
        SentimentExample(
            text="Credit costs increased significantly due to commercial real estate exposure.",
            label="negative",
            confidence=0.94
        ),
        SentimentExample(
            text="The company lowered full-year guidance citing macroeconomic uncertainty.",
            label="negative",
            confidence=0.91
        ),
        SentimentExample(
            text="Net charge-offs rose to their highest level in three years.",
            label="negative",
            confidence=0.89
        ),

        # Neutral examples
        SentimentExample(
            text="The company maintained its quarterly dividend of $0.50 per share.",
            label="neutral",
            confidence=0.85
        ),
        SentimentExample(
            text="Total assets remained relatively unchanged from the prior quarter.",
            label="neutral",
            confidence=0.82
        ),
        SentimentExample(
            text="The acquisition is expected to close in the third quarter.",
            label="neutral",
            confidence=0.80
        ),
        SentimentExample(
            text="Management reaffirmed existing guidance for the fiscal year.",
            label="neutral",
            confidence=0.78
        ),

        # More complex/nuanced examples
        SentimentExample(
            text="Despite revenue growth, profitability was impacted by higher operating costs.",
            label="negative",
            confidence=0.75
        ),
        SentimentExample(
            text="The restructuring program is expected to generate annual savings of $200 million.",
            label="positive",
            confidence=0.85
        ),
        SentimentExample(
            text="Market share gains offset pricing pressures in the competitive environment.",
            label="neutral",
            confidence=0.70
        )
    ]

    return samples


def get_sentiment_by_label(
    examples: List[SentimentExample],
    label: str
) -> List[SentimentExample]:
    """Filter examples by sentiment label"""
    return [ex for ex in examples if ex.label == label]


def get_balanced_sample(n_per_class: int = 5) -> List[SentimentExample]:
    """Get balanced sample with equal examples per class"""
    samples = get_sample_sentiment_examples()

    positive = [ex for ex in samples if ex.label == "positive"][:n_per_class]
    negative = [ex for ex in samples if ex.label == "negative"][:n_per_class]
    neutral = [ex for ex in samples if ex.label == "neutral"][:n_per_class]

    return positive + negative + neutral
