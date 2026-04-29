from dataclasses import dataclass, field

import numpy as np
import torch
from utils import TensorMetricAdapter

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import METRICS
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.base import BaseMetric
from chimera_ml.metrics.prf_metric import PRFMetric
from chimera_ml.metrics.regression_metric import MAEMetric


@dataclass
class AgeMAEMetric(BaseMetric):
    age_scale: float = 100.0
    _metric: TensorMetricAdapter = field(init=False)

    def __post_init__(self) -> None:
        self._metric = TensorMetricAdapter(MAEMetric())

    def reset(self) -> None:
        self._metric.reset()

    @torch.no_grad()
    def update(self, output: ModelOutput, batch: Batch) -> None:
        if batch.targets is None:
            return

        age_pred = torch.sigmoid((output.aux or {})["age"]) * self.age_scale
        age_target = batch.targets[:, 1].float() * self.age_scale
        self._metric.update(age_pred, age_target)

    def compute(self) -> dict[str, float]:
        values = self._metric.compute()
        return {"age_mae": values["mae"]} if values else {}


@dataclass
class GenderPRFMetric(BaseMetric):
    zero_division: int = 0
    _metric: TensorMetricAdapter = field(init=False)

    def __post_init__(self) -> None:
        self._metric = TensorMetricAdapter(PRFMetric(average="macro", zero_division=self.zero_division))

    def reset(self) -> None:
        self._metric.reset()

    @torch.no_grad()
    def update(self, output: ModelOutput, batch: Batch) -> None:
        if batch.targets is None:
            return

        gen_logits = (output.aux or {})["gen"]
        gen_target = batch.targets[:, 0].long()
        self._metric.update(gen_logits, gen_target)

    def compute(self) -> dict[str, float]:
        values = self._metric.compute()
        if not values:
            return {}

        return {
            "gen_precision": values["macro_precision"],
            "gen_uar": values["macro_recall"],
            "gen_macro_f1": values["macro_f1"],
        }


@dataclass
class MaskUARMetric(BaseMetric):
    zero_division: int = 0
    _metric: TensorMetricAdapter = field(init=False)

    def __post_init__(self) -> None:
        self._metric = TensorMetricAdapter(PRFMetric(average="macro", zero_division=self.zero_division))

    def reset(self) -> None:
        self._metric.reset()

    @torch.no_grad()
    def update(self, output: ModelOutput, batch: Batch) -> None:
        if batch.targets is None or "mask" not in (output.aux or {}) or batch.targets.shape[1] <= 2:
            return

        mask_logits = (output.aux or {})["mask"]
        mask_target = batch.targets[:, 2].long()
        self._metric.update(mask_logits, mask_target)

    def compute(self) -> dict[str, float]:
        values = self._metric.compute()
        return {"mask_uar": values["macro_recall"]} if values else {}


@dataclass
class AgePCCMetric(BaseMetric):
    age_scale: float = 100.0
    eps: float = 1e-8

    def reset(self) -> None:
        self._y_true: list[float] = []
        self._y_pred: list[float] = []

    @torch.no_grad()
    def update(self, output: ModelOutput, batch: Batch) -> None:
        if batch.targets is None:
            return

        age_pred = torch.sigmoid((output.aux or {})["age"]).detach().cpu().numpy() * self.age_scale
        age_true = batch.targets[:, 1].detach().cpu().float().numpy() * self.age_scale
        self._y_pred.extend(age_pred.astype(float).tolist())
        self._y_true.extend(age_true.astype(float).tolist())

    def compute(self) -> dict[str, float]:
        if len(self._y_true) < 2:
            return {}

        y_true = np.asarray(self._y_true, dtype=np.float64)
        y_pred = np.asarray(self._y_pred, dtype=np.float64)
        true_centered = y_true - np.mean(y_true)
        pred_centered = y_pred - np.mean(y_pred)
        denom = np.sqrt(np.sum(true_centered**2)) * np.sqrt(np.sum(pred_centered**2))
        value = 0.0 if denom < self.eps else float(np.sum(true_centered * pred_centered) / denom)
        return {"age_pcc": value}


@METRICS.register("age_mae_metric")
def age_mae_metric(**params):
    return AgeMAEMetric(**params)


@METRICS.register("age_pcc_metric")
def age_pcc_metric(**params):
    return AgePCCMetric(**params)


@METRICS.register("gender_prf_macro_metric")
def gender_prf_macro_metric(**params):
    return GenderPRFMetric(**params)


@METRICS.register("mask_uar_metric")
def mask_uar_metric(**params):
    return MaskUARMetric(**params)
