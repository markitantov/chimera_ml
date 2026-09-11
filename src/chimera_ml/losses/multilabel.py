import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss


class BCEWithLogitsLoss(BaseLoss):
    """Binary cross-entropy loss for multi-label logits and targets.

    Args:
        pos_weight: Optional positive-class weighting tensor.
        reduction: PyTorch reduction mode, typically mean, sum, or none.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None, reduction: str = "mean"):
        self._loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        return self._loss(output.preds, batch.targets.float())


@LOSSES.register("bce_with_logits_loss")
def bce_with_logits_loss(**params):
    return BCEWithLogitsLoss(**params)
