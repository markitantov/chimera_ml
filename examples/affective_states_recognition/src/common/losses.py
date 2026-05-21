from typing import Any

import torch
import torch.nn.functional as F

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss


def _prepare_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == logits.ndim and target.shape[-1] == logits.shape[-1]:
        return target.float()

    return target.long().view(-1)


class EmotionSentimentLoss(BaseLoss):
    def __init__(
        self,
        *,
        emotion_weights: list[float] | None = None,
        sentiment_weights: list[float] | None = None,
        emotion_alpha: float = 1.0,
        sentiment_alpha: float = 1.0,
    ) -> None:
        self.emotion_alpha = float(emotion_alpha)
        self.sentiment_alpha = float(sentiment_alpha)
        self.emotion_weights = (
            torch.as_tensor(emotion_weights, dtype=torch.float32) if emotion_weights is not None else None
        )
        self.sentiment_weights = (
            torch.as_tensor(sentiment_weights, dtype=torch.float32) if sentiment_weights is not None else None
        )

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        if batch.targets is None:
            raise ValueError("EmotionSentimentLoss requires targets.")

        aux = output.aux or {}
        emo_logits = aux["emo"]
        sen_logits = aux["sen"]

        emo_dim = int(emo_logits.shape[-1])
        emo_target = _prepare_target(emo_logits, batch.targets[:, :emo_dim])
        sen_target = _prepare_target(sen_logits, batch.targets[:, emo_dim:])

        emo_weights = self.emotion_weights.to(emo_logits.device) if self.emotion_weights is not None else None
        sen_weights = self.sentiment_weights.to(sen_logits.device) if self.sentiment_weights is not None else None

        emo_loss = F.cross_entropy(emo_logits, emo_target, weight=emo_weights)
        sen_loss = F.cross_entropy(sen_logits, sen_target, weight=sen_weights)
        return self.emotion_alpha * emo_loss + self.sentiment_alpha * sen_loss


@LOSSES.register("emotion_sentiment_loss")
def emotion_sentiment_loss(context: Any | None = None, **params: Any) -> EmotionSentimentLoss:
    emotion_weights = params.pop("emotion_weights", None)
    if emotion_weights is None and context is not None:
        emotion_weights = context.get("data.emotion_class_weights")

    sentiment_weights = params.pop("sentiment_weights", None)
    if sentiment_weights is None and context is not None:
        sentiment_weights = context.get("data.sentiment_class_weights")

    return EmotionSentimentLoss(
        emotion_weights=emotion_weights,
        sentiment_weights=sentiment_weights,
        **params,
    )
