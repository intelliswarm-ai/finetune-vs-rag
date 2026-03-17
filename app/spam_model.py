"""
DistilBERT Spam/Phishing Detection Model
Adapted from: github.com/intelliswarm-ai/enterprise-mailbox-assistant/model-fine-tuned-llm

Architecture: distilbert-base-uncased (66M parameters) with classification head.
Checkpoint contains model_state_dict + config dict.
"""
import re
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer


class DistilBertForPhishingDetection(nn.Module):
    """DistilBERT with a classification head for binary spam detection."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout_rate: float = 0.3,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.config = DistilBertConfig.from_pretrained(model_name)
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        self.pre_classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        pooled_output = hidden_state[:, 0]  # CLS token
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return {"loss": loss, "logits": logits}


class SpamDetector:
    """Inference wrapper for the fine-tuned DistilBERT spam detector."""

    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_length = 512

    def _load_model(self, checkpoint_path: str) -> DistilBertForPhishingDetection:
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)
        config = checkpoint["config"]
        model = DistilBertForPhishingDetection(
            model_name=config["model_name"],
            num_labels=config["num_labels"],
            dropout_rate=config["dropout_rate"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        return model

    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"http\S+", " [URL] ", text)
        text = re.sub(r"\S+@\S+", " [EMAIL] ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def predict_single(self, email_text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        cleaned_text = self.clean_text(email_text)
        encoding = self.tokenizer(
            cleaned_text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)
            prediction = torch.argmax(outputs["logits"], dim=-1)
        # label 0 = legitimate (ham), label 1 = phishing (spam)
        prediction_label = "spam" if prediction.item() == 1 else "ham"
        confidence = probs[0, prediction.item()].item()
        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "probabilities": {
                "ham": probs[0, 0].item(),
                "spam": probs[0, 1].item(),
            },
            "input_tokens": encoding["input_ids"].shape[1],
        }
