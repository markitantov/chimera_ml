import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss for class logits and integer class targets.

    Args:
        label_smoothing: Amount of uniform label smoothing passed to PyTorch.
    """

    def __init__(self, label_smoothing: float = 0.0):
        self._loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        return self._loss(output.preds, batch.targets)


@LOSSES.register("cross_entropy_loss")
def cross_entropy_loss(**params):
    return CrossEntropyLoss(**params)
