from dataclasses import dataclass
from typing import Any

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.base import BaseMetric


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

    def reset(self) -> None:
        self.metric.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # Keep tensors as-is; metric usually detaches/cpu's internally.
        out = ModelOutput(preds=preds)
        batch = Batch(inputs={}, targets=targets, meta={})
        self.metric.update(out, batch)

    def compute(self) -> dict[str, Any]:
        return self.metric.compute()
