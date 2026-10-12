import torch
from transformers import pipeline

from const import (
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_MAX_LENGTH,
    SENTIMENT_MODEL_NAME,
)


class SentimentEstimator:

    # Initialize the sentiment model on the available inference device
    # ---------------------------------------------------------------
    def __init__(self):
        self._device = 0 if torch.cuda.is_available() else -1
        self._classifier = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL_NAME,
            device=self._device,
            model_kwargs={"attn_implementation": "eager"},
        )

    # Estimate sentiment for a batch of texts
    # ---------------------------------------
    def get_sentiment(self, texts):
        results = self._classifier(
            texts,
            top_k=None,
            truncation=True,
            max_length=SENTIMENT_MAX_LENGTH,
            batch_size=SENTIMENT_BATCH_SIZE,
        )

        sentiments = []
        for text, scores in zip(texts, results):
            probabilities = {
                item["label"].lower(): item["score"]
                for item in scores
            }

            sentiment = (
                probabilities["positive"]
                - probabilities["negative"]
            )

            sentiments.append(
                {
                    "text": text,
                    "sentiment": sentiment,
                    "probabilities": probabilities,
                }
            )

        return sentiments

    # Return the device used for inference
    # ------------------------------------
    def get_device(self):
        if self._device == -1:
            return "cpu"

        return torch.cuda.get_device_name(self._device)
