from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.base import BaseMetric


def ccc_1d(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred, target: [B]
    Returns: 1 - CCC (so lower is better).
    """
    pred = pred.float()
    target = target.float()

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)

    cov = ((pred - pred_mean) * (target - target_mean)).mean()

    return (2.0 * cov) / (pred_var + target_var + (pred_mean - target_mean).pow(2) + eps)



@dataclass
class TensorMetricAdapter:
    """Adapter to feed raw (preds, targets) tensors into a metric that expects (ModelOutput, Batch).

    This avoids duplicating metric logic while keeping the metric API unchanged.

    Usage:
        adapter = TensorMetricAdapter(metric)
        adapter.reset()
        adapter.update(preds, targets)
        out = adapter.compute()
    """

    metric: BaseMetric
    device: torch.device | None = None

    def reset(self) -> None:
        self.metric.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # Keep tensors as-is; metric usually detaches/cpu's internally.
        out = ModelOutput(preds=preds)
        batch = Batch(inputs={}, targets=targets, meta={})
        self.metric.update(out, batch)

    def compute(self) -> dict[str, Any]:
        return self.metric.compute()