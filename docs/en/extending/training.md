# Custom losses and metrics

A custom loss implements `BaseLoss.__call__(output, batch) -> torch.Tensor` and is registered in `LOSSES`. It must return a differentiable scalar or a reduction compatible with the Trainer and should validate targets.

```python
import torch
from chimera_ml.core import Batch, LOSSES, METRICS, ModelOutput
from chimera_ml.losses.base import BaseLoss
from chimera_ml.metrics.base import BaseMetric

class L2Loss(BaseLoss):
    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction
    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        value = (output.preds - batch.targets).square()
        if self.reduction == "mean": return value.mean()
        if self.reduction == "sum": return value.sum()
        if self.reduction == "none": return value
        raise ValueError("reduction must be mean, sum, or none")

@LOSSES.register("my_l2_loss")
def my_l2_loss(reduction: str = "mean"):
    return L2Loss(reduction=reduction)

class MeanBiasMetric(BaseMetric):
    def reset(self): self._values = []
    def update(self, output: ModelOutput, batch: Batch):
        self._values.append((output.preds.detach() - batch.targets.detach()).mean().item())
    def compute(self) -> dict[str, float]:
        return {"bias": sum(self._values) / len(self._values)} if self._values else {}

@METRICS.register("mean_bias_metric")
def mean_bias_metric(): return MeanBiasMetric()
```

Configure them with `loss: {name: my_l2_loss, params: {reduction: mean}}` and `metrics: [{name: mean_bias_metric, params: {}}]`. Trainer resets metrics per split/epoch, updates labeled batches, and prefixes returned names: `{"bias": 0.2}` becomes `val/bias`. Use that exact key for checkpoint, early stopping, scheduler-monitor, and sweep callbacks.
