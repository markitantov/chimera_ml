from dataclasses import dataclass

import numpy as np
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import METRICS
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics._utils import compute_confusion_matrix, normalize_confusion_matrix
from chimera_ml.metrics.base import BaseMetric


@dataclass
class ConfusionMatrixMetric(BaseMetric):
    """Confusion matrix metric.

    Accumulates y_true/y_pred; compute() builds the matrix and returns cm_acc.
    """

    normalize: str | None = None  # None | 'true' | 'pred' | 'all'

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._y_true = []
        self._y_pred = []
        self._cm = None

    @torch.no_grad()
    def update(self, output: ModelOutput, batch: Batch) -> None:
        y_pred = output.preds.argmax(dim=-1).view(-1).detach().cpu().numpy()
        y_true = batch.targets.view(-1).detach().cpu().numpy()
        self._y_pred.extend(y_pred.tolist())
        self._y_true.extend(y_true.tolist())

    def compute(self) -> dict[str, float]:
        if not self._y_true:
            return {}

        y_true = np.asarray(self._y_true)
        y_pred = np.asarray(self._y_pred)

        cm = compute_confusion_matrix(y_true=y_true, y_pred=y_pred)
        if self.normalize is None:
            self._cm = cm
        else:
            self._cm = normalize_confusion_matrix(cm=cm, normalize=self.normalize)

        acc = float(np.trace(self._cm) / max(np.sum(self._cm), 1.0))
        return {"cm_acc": acc}

    def value(self) -> np.ndarray | None:
        return self._cm


@METRICS.register("confusion_matrix_metric")
def confusion_matrix_metric(**params):
    return ConfusionMatrixMetric(**params)
