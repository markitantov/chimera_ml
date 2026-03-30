from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import METRICS
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.base import BaseMetric

Multioutput = Literal["raw_values", "uniform_average", "variance_weighted"]


@dataclass
class _SklearnRegressionBaseMetric(BaseMetric):
    """Base class for sklearn regression metrics.

    Accumulates y_true and y_pred during epoch, then computes metric at epoch end.
    Flattens tensors to (N, D) where D is output dimension.
    """

    multioutput: Multioutput = "uniform_average"

    def reset(self) -> None:
        self._y_true = []
        self._y_pred = []

    @torch.no_grad()
    def update(self, output: ModelOutput, batch: Batch) -> None:
        y_pred = output.preds.detach()
        y_true = batch.targets.detach()

        # Flatten to (B, D)
        y_pred = y_pred.view(-1, 1) if y_pred.ndim == 1 else y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(-1, 1) if y_true.ndim == 1 else y_true.view(y_true.shape[0], -1)

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Regression metric: preds shape {tuple(y_pred.shape)} must match targets shape {tuple(y_true.shape)}"
            )

        self._y_pred.append(y_pred.cpu().numpy())
        self._y_true.append(y_true.cpu().numpy())

    def _stack(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._y_true:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

        y_true = np.concatenate(self._y_true, axis=0)
        y_pred = np.concatenate(self._y_pred, axis=0)
        return y_true, y_pred


@dataclass
class SklearnMAEMetric(_SklearnRegressionBaseMetric):
    def compute(self) -> dict[str, float]:
        y_true, y_pred = self._stack()
        if y_true.shape[0] == 0:
            return {}

        v = mean_absolute_error(y_true, y_pred, multioutput=self.multioutput)
        return {"mae": float(v)}


@dataclass
class SklearnRMSEMetric(_SklearnRegressionBaseMetric):
    def compute(self) -> dict[str, float]:
        y_true, y_pred = self._stack()
        if y_true.shape[0] == 0:
            return {}

        v = root_mean_squared_error(y_true, y_pred, multioutput=self.multioutput)
        return {"rmse": v}


@dataclass
class SklearnMSEMetric(_SklearnRegressionBaseMetric):
    def compute(self) -> dict[str, float]:
        y_true, y_pred = self._stack()
        if y_true.shape[0] == 0:
            return {}

        v = mean_squared_error(y_true, y_pred, multioutput=self.multioutput)
        return {"mse": v}


@dataclass
class SklearnR2Metric(_SklearnRegressionBaseMetric):
    def compute(self) -> dict[str, float]:
        y_true, y_pred = self._stack()
        if y_true.shape[0] == 0:
            return {}

        v = r2_score(y_true, y_pred, multioutput=self.multioutput)
        return {"r2": float(v)}


@METRICS.register("mae_metric")
def mae_metric(**params):
    return SklearnMAEMetric(**params)


@METRICS.register("mse_metric")
def mse_metric(**params):
    return SklearnMSEMetric(**params)


@METRICS.register("rmse_metric")
def rmse_metric(**params):
    return SklearnRMSEMetric(**params)


@METRICS.register("r2_metric")
def r2_metric(**params):
    return SklearnR2Metric(**params)
