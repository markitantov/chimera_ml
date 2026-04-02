from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import METRICS
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.base import BaseMetric


def _validate_multioutput(
    multioutput: Literal["raw_values", "uniform_average", "variance_weighted"], allowed: tuple[str, ...]
) -> None:
    if multioutput not in allowed:
        allowed_values = ", ".join(f"'{v}'" for v in allowed)
        raise ValueError(f"Invalid multioutput='{multioutput}'. Allowed values for this metric: {allowed_values}.")


def _aggregate(
    values: np.ndarray,
    multioutput: Literal["raw_values", "uniform_average", "variance_weighted"],
    *,
    variance_weights: np.ndarray | None = None,
):
    if multioutput == "raw_values":
        return values
    if multioutput == "uniform_average":
        return float(np.mean(values))
    if multioutput == "variance_weighted":
        if variance_weights is None:
            raise ValueError("multioutput='variance_weighted' is only supported for r2_metric.")
        total_weight = float(np.sum(variance_weights))
        if total_weight == 0.0:
            return float(np.mean(values))
        return float(np.average(values, weights=variance_weights))
    raise ValueError(f"Unknown multioutput='{multioutput}'.")


def _metric_output(value):
    if isinstance(value, np.ndarray):
        return value
    return float(value)


@dataclass
class RegressionBaseMetric(BaseMetric):
    """Base class for regression metrics.

    Accumulates y_true and y_pred during epoch, then computes metric at epoch end.
    Flattens tensors to (N, D) where D is output dimension.
    """

    multioutput: Literal["raw_values", "uniform_average", "variance_weighted"] = "uniform_average"

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
class MAEMetric(RegressionBaseMetric):
    def compute(self) -> dict[str, float]:
        y_true, y_pred = self._stack()
        if y_true.shape[0] == 0:
            return {}

        _validate_multioutput(self.multioutput, allowed=("raw_values", "uniform_average"))

        errors = np.abs(y_true - y_pred)
        raw = np.mean(errors, axis=0)
        v = _aggregate(raw, self.multioutput)
        return {"mae": _metric_output(v)}


@dataclass
class RMSEMetric(RegressionBaseMetric):
    def compute(self) -> dict[str, float]:
        y_true, y_pred = self._stack()
        if y_true.shape[0] == 0:
            return {}

        _validate_multioutput(self.multioutput, allowed=("raw_values", "uniform_average"))

        errors = y_true - y_pred
        mse_raw = np.mean(np.square(errors), axis=0)
        rmse_raw = np.sqrt(mse_raw)
        v = _aggregate(rmse_raw, self.multioutput)
        return {"rmse": _metric_output(v)}


@dataclass
class MSEMetric(RegressionBaseMetric):
    def compute(self) -> dict[str, float]:
        y_true, y_pred = self._stack()
        if y_true.shape[0] == 0:
            return {}

        _validate_multioutput(self.multioutput, allowed=("raw_values", "uniform_average"))

        errors = y_true - y_pred
        mse_raw = np.mean(np.square(errors), axis=0)
        v = _aggregate(mse_raw, self.multioutput)
        return {"mse": _metric_output(v)}


@dataclass
class R2Metric(RegressionBaseMetric):
    def compute(self) -> dict[str, float]:
        y_true, y_pred = self._stack()
        if y_true.shape[0] == 0:
            return {}

        if y_true.shape[0] < 2:
            raw = np.full((y_true.shape[1],), np.nan, dtype=np.float64)
            v = _aggregate(raw, self.multioutput)
            return {"r2": _metric_output(v)}

        ss_res = np.sum(np.square(y_true - y_pred), axis=0, dtype=np.float64)
        y_mean = np.mean(y_true, axis=0, dtype=np.float64)
        ss_tot = np.sum(np.square(y_true - y_mean), axis=0, dtype=np.float64)

        raw = np.empty_like(ss_res, dtype=np.float64)
        den_nonzero = ss_tot != 0.0
        raw[den_nonzero] = 1.0 - (ss_res[den_nonzero] / ss_tot[den_nonzero])

        den_zero = ~den_nonzero
        raw[den_zero] = np.where(ss_res[den_zero] == 0.0, 1.0, 0.0)

        _validate_multioutput(self.multioutput, allowed=("raw_values", "uniform_average", "variance_weighted"))
        v = _aggregate(raw, self.multioutput, variance_weights=ss_tot)
        return {"r2": _metric_output(v)}


@METRICS.register("mae_metric")
def mae_metric(**params):
    return MAEMetric(**params)


@METRICS.register("mse_metric")
def mse_metric(**params):
    return MSEMetric(**params)


@METRICS.register("rmse_metric")
def rmse_metric(**params):
    return RMSEMetric(**params)


@METRICS.register("r2_metric")
def r2_metric(**params):
    return R2Metric(**params)
