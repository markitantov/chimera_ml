from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from common.utils import classification_metrics, cmu_accuracy
from utils import TensorMetricAdapter

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import METRICS
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.base import BaseMetric
from chimera_ml.metrics.prf_metric import PRFMetric


def _targets_to_labels(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1:
        return targets.long()

    return targets.argmax(dim=-1).long()


def _cmu_targets_and_predicts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    target_arr = targets.detach().cpu().numpy()

    if probs.shape[1] > 6:
        target_arr = (target_arr > 0).astype(int)[:, 1:]
        neutral_threshold = 1.0 - 1.0 / probs.shape[1]
        other_threshold = 1.0 / probs.shape[1]
        neutral_mask = probs[:, 0] >= neutral_threshold
        other_mask = probs[:, 1:] >= other_threshold
        predicts = np.zeros_like(probs[:, 1:], dtype=int)
        predicts[~neutral_mask] = other_mask[~neutral_mask].astype(int)
        return target_arr, predicts

    target_arr = (target_arr > 0).astype(int)
    predicts = (probs >= 0.5).astype(int)
    return target_arr, predicts


def _sample_meta(batch: Batch) -> list[dict[str, Any]]:
    if not batch.meta or not isinstance(batch.meta.get("sample_meta"), list):
        size = int(batch.targets.shape[0]) if batch.targets is not None else 0
        return [{} for _ in range(size)]

    return [item if isinstance(item, dict) else {} for item in batch.meta["sample_meta"]]


@dataclass
class EmotionMetric(BaseMetric):
    zero_division: int = 0
    _macro_metric: TensorMetricAdapter = field(init=False)
    _weighted_metric: TensorMetricAdapter = field(init=False)

    def __post_init__(self) -> None:
        self._macro_metric = TensorMetricAdapter(PRFMetric(average="macro", zero_division=self.zero_division))
        self._weighted_metric = TensorMetricAdapter(PRFMetric(average="weighted", zero_division=self.zero_division))
        self.reset()

    def reset(self) -> None:
        self._macro_metric.reset()
        self._weighted_metric.reset()
        self._classification_count = 0
        self._cmu_targets: list[np.ndarray] = []
        self._cmu_predicts: list[np.ndarray] = []
        self._cmu_count = 0

    @torch.no_grad()
    def update(self, output: ModelOutput, batch: Batch) -> None:
        if batch.targets is None:
            return

        emo_logits = (output.aux or {})["emo"]
        emo_dim = int(emo_logits.shape[-1])
        emo_targets = batch.targets[:, :emo_dim]
        metas = _sample_meta(batch)

        cmu_mask = torch.tensor(
            ["CMUMOSEI" in str(meta.get("corpus_name", "")) for meta in metas],
            device=emo_logits.device,
            dtype=torch.bool,
        )
        other_mask = ~cmu_mask

        if torch.any(other_mask):
            other_logits = emo_logits[other_mask]
            other_targets = _targets_to_labels(emo_targets[other_mask])
            self._macro_metric.update(other_logits, other_targets)
            self._weighted_metric.update(other_logits, other_targets)
            self._classification_count += int(other_mask.sum().item())

        if torch.any(cmu_mask):
            cmu_logits = emo_logits[cmu_mask]
            cmu_targets = emo_targets[cmu_mask]
            t_arr, p_arr = _cmu_targets_and_predicts(cmu_logits, cmu_targets)
            self._cmu_targets.append(t_arr)
            self._cmu_predicts.append(p_arr)
            self._cmu_count += int(cmu_mask.sum().item())

    def compute(self) -> dict[str, float]:
        values: dict[str, float] = {}
        if self._classification_count:
            macro = self._macro_metric.compute()
            weighted = self._weighted_metric.compute()
            values = {
                "emo_macro_f1": float(macro.get("macro_f1", 0.0)) * 100.0,
                "emo_uar": float(macro.get("macro_recall", 0.0)) * 100.0,
                "emo_weighted_f1": float(weighted.get("weighted_f1", 0.0)) * 100.0,
                "emo_war": float(weighted.get("weighted_recall", 0.0)) * 100.0,
            }

        if self._cmu_count:
            targets = np.concatenate(self._cmu_targets, axis=0)
            predicts = np.concatenate(self._cmu_predicts, axis=0)
            mwa = []
            macro_f1 = []
            weighted_f1 = []
            uar = []
            war = []
            for index in range(predicts.shape[1]):
                target_col = targets[:, index].astype(int)
                pred_col = predicts[:, index].astype(int)
                mwa.append(cmu_accuracy(target_col, pred_col) * 100.0)
                scores = classification_metrics(target_col, pred_col)
                macro_f1.append(scores["macro_f1"])
                weighted_f1.append(scores["weighted_f1"])
                uar.append(scores["uar"])
                war.append(scores["war"])

            cmu_values = {
                "emo_macro_f1": float(np.mean(macro_f1)),
                "emo_uar": float(np.mean(uar)),
                "emo_weighted_f1": float(np.mean(weighted_f1)),
                "emo_war": float(np.mean(war)),
                "emo_mwa": float(np.mean(mwa)),
            }
            if not values:
                return cmu_values

            cls_weight = float(self._classification_count)
            cmu_weight = float(self._cmu_count)
            total = cls_weight + cmu_weight
            for key in ("emo_macro_f1", "emo_uar", "emo_weighted_f1", "emo_war"):
                values[key] = (values[key] * cls_weight + cmu_values[key] * cmu_weight) / total

            values["emo_mwa"] = cmu_values["emo_mwa"]

        return values


@dataclass
class SentimentMetric(BaseMetric):
    zero_division: int = 0
    _macro_metric: TensorMetricAdapter = field(init=False)
    _weighted_metric: TensorMetricAdapter = field(init=False)

    def __post_init__(self) -> None:
        self._macro_metric = TensorMetricAdapter(PRFMetric(average="macro", zero_division=self.zero_division))
        self._weighted_metric = TensorMetricAdapter(PRFMetric(average="weighted", zero_division=self.zero_division))
        self.reset()

    def reset(self) -> None:
        self._macro_metric.reset()
        self._weighted_metric.reset()

    @torch.no_grad()
    def update(self, output: ModelOutput, batch: Batch) -> None:
        if batch.targets is None:
            return

        sen_logits = (output.aux or {})["sen"]
        emo_dim = int((output.aux or {})["emo"].shape[-1])
        sen_targets = _targets_to_labels(batch.targets[:, emo_dim:])
        self._macro_metric.update(sen_logits, sen_targets)
        self._weighted_metric.update(sen_logits, sen_targets)

    def compute(self) -> dict[str, float]:
        macro = self._macro_metric.compute()
        weighted = self._weighted_metric.compute()
        if not macro and not weighted:
            return {}

        return {
            "sen_macro_f1": float(macro.get("macro_f1", 0.0)) * 100.0,
            "sen_uar": float(macro.get("macro_recall", 0.0)) * 100.0,
            "sen_weighted_f1": float(weighted.get("weighted_f1", 0.0)) * 100.0,
            "sen_war": float(weighted.get("weighted_recall", 0.0)) * 100.0,
        }


@dataclass
class EmotionSentimentCombinedMetric(BaseMetric):
    zero_division: int = 0
    _emotion_metric: EmotionMetric = field(init=False)
    _sentiment_metric: SentimentMetric = field(init=False)

    def __post_init__(self) -> None:
        self._emotion_metric = EmotionMetric(zero_division=self.zero_division)
        self._sentiment_metric = SentimentMetric(zero_division=self.zero_division)
        self.reset()

    def reset(self) -> None:
        self._emotion_metric.reset()
        self._sentiment_metric.reset()

    @torch.no_grad()
    def update(self, output: ModelOutput, batch: Batch) -> None:
        self._emotion_metric.update(output, batch)
        self._sentiment_metric.update(output, batch)

    def compute(self) -> dict[str, float]:
        emotion = self._emotion_metric.compute()
        sentiment = self._sentiment_metric.compute()
        if not emotion or not sentiment:
            return {}

        return {"emo_sen_combined": (emotion["emo_macro_f1"] + sentiment["sen_macro_f1"]) / 2.0}


@METRICS.register("emotion_metric")
def emotion_metric(**params: Any) -> EmotionMetric:
    return EmotionMetric(**params)


@METRICS.register("sentiment_metric")
def sentiment_metric(**params: Any) -> SentimentMetric:
    return SentimentMetric(**params)


@METRICS.register("emotion_sentiment_combined_metric")
def emotion_sentiment_combined_metric(**params: Any) -> EmotionSentimentCombinedMetric:
    return EmotionSentimentCombinedMetric(**params)
