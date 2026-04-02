from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import METRICS
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics._utils import compute_confusion_matrix
from chimera_ml.metrics.base import BaseMetric


def _safe_divide(num: np.ndarray, den: np.ndarray, zero_division: float) -> np.ndarray:
    out = np.full(num.shape, zero_division, dtype=np.float64)
    mask = den != 0.0
    out[mask] = num[mask] / den[mask]
    return out


def _safe_f1(precision: np.ndarray, recall: np.ndarray, zero_division: float) -> np.ndarray:
    den = precision + recall
    out = np.full(precision.shape, zero_division, dtype=np.float64)
    mask = den != 0.0
    out[mask] = 2.0 * precision[mask] * recall[mask] / den[mask]
    return out


@dataclass
class PRFMetric(BaseMetric):
    """Precision / Recall / F1.

    Accumulates y_true and y_pred during epoch.
    Computes metrics at epoch end.
    """

    average: Literal["micro", "macro", "weighted"]
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

        y_true = np.asarray(self._y_true)
        y_pred = np.asarray(self._y_pred)
        cm = compute_confusion_matrix(y_true=y_true, y_pred=y_pred).astype(np.float64)

        tp = np.diag(cm)
        predicted_positive = cm.sum(axis=0)
        actual_positive = cm.sum(axis=1)
        zero_division = float(self.zero_division)

        precision_per_class = _safe_divide(tp, predicted_positive, zero_division)
        recall_per_class = _safe_divide(tp, actual_positive, zero_division)
        f1_per_class = _safe_f1(precision_per_class, recall_per_class, zero_division)

        if self.average == "micro":
            tp_sum = float(tp.sum())
            predicted_sum = float(predicted_positive.sum())
            actual_sum = float(actual_positive.sum())

            precision = tp_sum / predicted_sum if predicted_sum != 0.0 else zero_division
            recall = tp_sum / actual_sum if actual_sum != 0.0 else zero_division
            den = precision + recall
            f1 = (2.0 * precision * recall / den) if den != 0.0 else zero_division
        elif self.average == "macro":
            precision = float(np.mean(precision_per_class))
            recall = float(np.mean(recall_per_class))
            f1 = float(np.mean(f1_per_class))
        elif self.average == "weighted":
            support = actual_positive
            total_support = float(support.sum())
            if total_support == 0.0:
                precision = zero_division
                recall = zero_division
                f1 = zero_division
            else:
                precision = float(np.average(precision_per_class, weights=support))
                recall = float(np.average(recall_per_class, weights=support))
                f1 = float(np.average(f1_per_class, weights=support))
        else:
            raise ValueError(f"Invalid average='{self.average}'. Expected one of: micro, macro, weighted.")

        prefix = self.average
        return {
            f"{prefix}_precision": float(precision),
            f"{prefix}_recall": float(recall),
            f"{prefix}_f1": float(f1),
        }


@METRICS.register("prf_macro_metric")
def prf_macro_metric(**params):
    return PRFMetric(average="macro", **params)


@METRICS.register("prf_micro_metric")
def prf_micro_metric(**params):
    return PRFMetric(average="micro", **params)


@METRICS.register("prf_weighted_metric")
def prf_weighted_metric(**params):
    return PRFMetric(average="weighted", **params)
