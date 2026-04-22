import torch
import torch.nn as nn
import torch.nn.functional as F

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss


class CCCLoss(nn.Module):
    """Lin's Concordance Correlation Coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Measures the agreement between two variables
    
    It is a product of
    - precision (pearson correlation coefficient) and
    - accuracy (closeness to 45 degree line)
    
    Interpretation
    - rho =  1: perfect agreement
    - rho =  0: no agreement
    - rho = -1: perfect disagreement
    
    Args:
        eps (float, optional): Avoiding division by zero. Defaults to 1e-8.
    """
    
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes CCC loss

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: 1 - CCC loss value
        """
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps
        )
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x, correction=0)
        y_s = torch.std(y, correction=0)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + self.eps)
        return 1 - ccc


class AGenderLoss(BaseLoss):
    def __init__(
        self,
        gender_weights: list[float] | None = None,
        mask_weights: list[float] | None = None,
        gender_alpha: float = 1.0,
        age_alpha: float = 1.0,
        mask_alpha: float = 0.0,
    ) -> None:
        self.gender_alpha = float(gender_alpha)
        self.age_alpha = float(age_alpha)
        self.mask_alpha = float(mask_alpha)
        self.gender_weights = torch.tensor(gender_weights, dtype=torch.float32) if gender_weights is not None else None
        self.mask_weights = torch.tensor(mask_weights, dtype=torch.float32) if mask_weights is not None else None
        self.age_loss = CCCLoss()

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        if batch.targets is None:
            raise ValueError("AGenderLoss requires targets.")

        preds = output.aux or {}
        gen_logits = preds["gen"]
        age_logits = preds["age"]
        gen_target = batch.targets[:, 0].long()
        age_target = batch.targets[:, 1].float()

        gender_weights = self.gender_weights.to(gen_logits.device) if self.gender_weights is not None else None
        loss = self.gender_alpha * F.cross_entropy(gen_logits, gen_target, weight=gender_weights)
        loss = loss + self.age_alpha * self.age_loss(torch.sigmoid(age_logits), age_target)

        if self.mask_alpha > 0.0 and "mask" in preds and batch.targets.shape[1] > 2:
            mask_logits = preds["mask"]
            mask_target = batch.targets[:, 2].long()
            mask_weights = self.mask_weights.to(mask_logits.device) if self.mask_weights is not None else None
            loss = loss + self.mask_alpha * F.cross_entropy(mask_logits, mask_target, weight=mask_weights)

        return loss


@LOSSES.register("agender_loss")
def agender_loss(**params):
    return AGenderLoss(**params)
