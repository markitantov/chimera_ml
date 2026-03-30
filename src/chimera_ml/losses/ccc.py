import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import LOSSES
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss


def _ccc_1d_rho(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
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
        x (torch.Tensor): Input tensor
        y (torch.Tensor): Target tensor
        eps (float, optional): Avoiding division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: CCC value
    Notes:
    - Uses population std/var (unbiased=False) for stability/consistency.
    - Adds eps to both rho denominator and final CCC denominator for numerical stability.
    """
    x = x.float()
    y = y.float()

    x_m = x.mean()
    y_m = y.mean()

    vx = x - x_m
    vy = y - y_m

    # Pearson correlation (rho)
    denom_rho = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + eps
    rho = torch.sum(vx * vy) / denom_rho

    x_s = x.std(unbiased=False)
    y_s = y.std(unbiased=False)

    return (2.0 * rho * x_s * y_s) / (x_s**2 + y_s**2 + (x_m - y_m) ** 2 + eps)


class CCCLoss(BaseLoss):
    """Concordance Correlation Coefficient loss for regression.

    Returns:
      loss = 1 - mean(CCC) across output dimensions.

    Expects:
      output.preds and batch.targets to have the same shape:
        (B,), (B, D) or (B, ...) which will be flattened to (B, Dflat).
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        pred = output.preds
        target = batch.targets

        pred = pred.view(-1, 1) if pred.ndim == 1 else pred.view(pred.shape[0], -1)
        target = target.view(-1, 1) if target.ndim == 1 else target.view(target.shape[0], -1)

        if pred.shape != target.shape:
            raise ValueError(
                f"CCCLoss: preds shape {tuple(pred.shape)} must match targets shape {tuple(target.shape)}"
            )

        ccc_vals = []
        for d in range(pred.shape[1]):
            ccc_vals.append(_ccc_1d_rho(pred[:, d], target[:, d], eps=self.eps))

        ccc = torch.stack(ccc_vals).mean()
        return 1.0 - ccc


@LOSSES.register("ccc_loss")
def ccc_loss(**params):
    return CCCLoss(**params)
