from typing import Optional, Union

import torch
import torch.nn.functional as F

from chimera_ml.losses.base import BaseLoss
from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.core.registry import LOSSES


class FocalLoss(BaseLoss):
    """Multi-class focal loss for logits + class indices."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, list[float]]] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor(float(alpha))
        else:
            self.alpha = torch.tensor([float(a) for a in alpha])

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        logits = output.preds
        targets = batch.targets.long()

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        idx = targets.view(-1, 1)
        log_pt = log_probs.gather(dim=-1, index=idx).squeeze(-1)
        pt = probs.gather(dim=-1, index=idx).squeeze(-1)

        if self.label_smoothing > 0.0:
            C = logits.shape[-1]
            smooth = self.label_smoothing
            pt = pt * (1.0 - smooth) + (smooth / C)

        focal = (1.0 - pt).clamp(min=0.0) ** self.gamma
        loss = -focal * log_pt

        if self.alpha is not None:
            if self.alpha.numel() == 1:
                loss = loss * self.alpha.to(loss.device)
            else:
                a = self.alpha.to(loss.device).gather(0, targets)
                loss = loss * a

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction: {self.reduction}")


@LOSSES.register("focal_loss")
def focal_loss(**params):
    return FocalLoss(**params)
