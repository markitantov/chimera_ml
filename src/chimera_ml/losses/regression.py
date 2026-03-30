import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss


class MSELoss(BaseLoss):
    """MSE for regression. Expects output.preds same shape as targets."""

    def __init__(self, reduction: str = "mean"):
        self._loss = nn.MSELoss(reduction=reduction)

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        return self._loss(output.preds, batch.targets)


class MAELoss(BaseLoss):
    """MAE (L1) for regression."""

    def __init__(self, reduction: str = "mean"):
        self._loss = nn.L1Loss(reduction=reduction)

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        return self._loss(output.preds, batch.targets)


@LOSSES.register("mse_loss")
def mse_loss(**params):
    return MSELoss(**params)


@LOSSES.register("mae_loss")
def mae_loss(**params):
    return MAELoss(**params)
