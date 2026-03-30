from dataclasses import dataclass
from typing import Literal

import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import METRICS
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.base import BaseMetric

Averaging = Literal["micro", "macro", "weighted"]


@dataclass
class SklearnPRFMetric(BaseMetric):
    """Precision / Recall / F1 via sklearn.

    Accumulates y_true and y_pred during epoch.
    Computes metrics at epoch end.
    """

    average: Averaging
    zero_division: int = 0

    def reset(self) -> None:
        self._y_true = []
        self._y_pred = []

    @torch.no_grad()
    def update(self, output: ModelOutput, batch: Batch) -> None:
        preds = output.preds.argmax(dim=-1).view(-1).cpu().numpy()
        targets = batch.targets.view(-1).cpu().numpy()

        self._y_pred.extend(preds.tolist())
        self._y_true.extend(targets.tolist())

    def compute(self) -> dict[str, float]:
        if not self._y_true:
            return {}

        precision = precision_score(
            self._y_true,
            self._y_pred,
            average=self.average,
            zero_division=self.zero_division,
        )
        recall = recall_score(
            self._y_true,
            self._y_pred,
            average=self.average,
            zero_division=self.zero_division,
        )
        f1 = f1_score(
            self._y_true,
            self._y_pred,
            average=self.average,
            zero_division=self.zero_division,
        )

        prefix = self.average
        return {
            f"{prefix}_precision": float(precision),
            f"{prefix}_recall": float(recall),
            f"{prefix}_f1": float(f1),
        }


@METRICS.register("prf_macro_metric")
def prf_macro_metric(**params):
    return SklearnPRFMetric(average="macro", **params)


@METRICS.register("prf_micro_metric")
def prf_micro_metric(**params):
    return SklearnPRFMetric(average="micro", **params)


@METRICS.register("prf_weighted_metric")
def prf_weighted_metric(**params):
    return SklearnPRFMetric(average="weighted", **params)
