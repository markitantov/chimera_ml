from dataclasses import dataclass, field
from typing import Any

import torch
import matplotlib.pyplot as plt
import numpy as np

from chimera_ml.callbacks._utils import resolve_splits
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.callbacks.plot_confusion_matrix_callback import (
    _fig_to_png_bytes,
    _plot_confusion_matrix,
)
from chimera_ml.core.registry import CALLBACKS
from chimera_ml.training.cached_split_outputs import CachedSplitOutputs


def _as_chunks(value: torch.Tensor | list[torch.Tensor] | None) -> list[torch.Tensor]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return [value.detach().cpu()]
    return [chunk.detach().cpu() for chunk in value if torch.is_tensor(chunk)]


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    if np.any(valid):
        np.add.at(cm, (y_true[valid], y_pred[valid]), 1)
    return cm


@dataclass
class GroupingCallback(BaseCallback):
    splits: list[str] = field(default_factory=lambda: ["val"])
    gender_class_names: list[str] | None = None
    age_scale: float = 100.0
    metric_prefix: str = "grouped"
    cc_metric_name: str = "all" # cross-corpus metric name
    log_confusion_matrix: bool = True
    artifact_path: str = "figures"
    filename_template: str = "confusion_matrix_epoch_{epoch}.png"
    title_template: str = "{corpus}. Confusion Matrix (epoch {epoch})"

    def __post_init__(self) -> None:
        self.gender_class_names = [str(name) for name in (self.gender_class_names or ["female", "male"])]
        self.gender_num_classes = len(self.gender_class_names)

    def on_fit_start(self, trainer: Any) -> None:
        trainer.config.collect_cache = True

    @torch.no_grad()
    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        logger = getattr(trainer, "mlflow_logger", None)
        grouped_by_split: dict[str, dict[str, list[dict[str, float | int]]]] = {}

        for split_name, _ in resolve_splits(trainer, self.splits):
            cached = trainer.get_cached_split_outputs(split_name)
            if cached is None or cached.targets is None:
                continue

            split_group = split_name.split("_", 1)[0]
            grouped = self._group_data(cached)
            for corpus_name, rows in grouped.items():
                grouped_by_split.setdefault(split_group, {}).setdefault(corpus_name, []).extend(rows)

        for split_group, grouped_all in grouped_by_split.items():
            all_rows: list[dict[str, float | int]] = []
            for corpus_name, rows in grouped_all.items():
                all_rows.extend(rows)
                metrics = self._compute_metrics(rows)
                if not metrics:
                    continue

                metric_scope = f"{split_group}_{corpus_name}_{self.metric_prefix}"
                grouped_metrics = {f"{metric_scope}/{key}": value for key, value in metrics.items()}
                logs.update(grouped_metrics)

                if logger is not None:
                    logger.log_metrics(grouped_metrics, step=epoch)
                    fig = _plot_confusion_matrix(
                        cm=self._compute_confusion_matrix(rows),
                        labels=self.gender_class_names,
                        title=self.title_template.format(corpus=corpus_name, epoch=epoch),
                    )
                    logger.log_artifact_bytes(
                        _fig_to_png_bytes(fig),
                        artifact_path=f"{self.artifact_path}/{metric_scope}",
                        filename=self.filename_template.format(corpus=corpus_name, epoch=epoch),
                    )
                    plt.close(fig)

                self._info(
                    trainer,
                    f"[GroupingCallback] "
                    + " | ".join(
                        f"{metric_scope}/{key}={value:.4f}"
                        for key, value in metrics.items()
                        if key != "num_files"
                    ),
                )

            if all_rows:
                metric_scope = f"{split_group}_{self.cc_metric_name.upper()}_{self.metric_prefix}"
                metrics = self._compute_metrics(all_rows)
                grouped_metrics = {f"{metric_scope}/{key}": value for key, value in metrics.items()}
                logs.update(grouped_metrics)

                if logger is not None:
                    logger.log_metrics(grouped_metrics, step=epoch)

                self._info(
                    trainer,
                    f"[GroupingCallback] "
                    + " | ".join(
                        f"{metric_scope}/{key}={value:.4f}"
                        for key, value in metrics.items()
                        if key != "num_files"
                    ),
                )

    def _group_data(self, cached: CachedSplitOutputs) -> dict[str, list[dict[str, float | int]]]:
        pred_chunks = _as_chunks(cached.preds)
        target_chunks = _as_chunks(cached.targets)

        sample_meta = list(cached.sample_meta or [])
        meta_index = 0
        grouped: dict[tuple[str, str], dict[str, Any]] = {}

        for pred_chunk, target_chunk in zip(pred_chunks, target_chunks, strict=False):
            bs = pred_chunk.shape[0]
            chunk_meta = sample_meta[meta_index : meta_index + bs]

            gen_logits = pred_chunk[:bs, : self.gender_num_classes]
            gen_probs = torch.softmax(gen_logits, dim=-1).numpy()
            age_preds = torch.sigmoid(pred_chunk[:bs, self.gender_num_classes]).numpy() * self.age_scale
            gen_targets = target_chunk[:bs, 0].long().numpy()
            age_targets = target_chunk[:bs, 1].float().numpy() * self.age_scale

            for idx in range(bs):
                meta = chunk_meta[idx]
                corpus_name = str(meta["corpus_name"])
                filename = str(meta["filename"])
                key = (corpus_name, filename)

                if key not in grouped:
                    grouped[key] = {
                        "gen_probs": [],
                        "age_preds": [],
                        "gen_target": int(gen_targets[idx]),
                        "age_target": float(age_targets[idx]),
                    }

                bucket = grouped[key]
                if bucket["gen_target"] != int(gen_targets[idx]):
                    raise ValueError(
                        f"[GroupingCallback] Inconsistent gen_target for {corpus_name}/{filename}: "
                        f"{bucket['gen_target']} != {int(gen_targets[idx])}"
                    )
                
                if bucket["age_target"] != float(age_targets[idx]):
                    raise ValueError(
                        f"[GroupingCallback] Inconsistent age_target for {corpus_name}/{filename}: "
                        f"{bucket['age_target']} != {float(age_targets[idx])}"
                    )
                
                bucket["gen_probs"].append(gen_probs[idx])
                bucket["age_preds"].append(float(age_preds[idx]))

            meta_index += bs

        per_corpus: dict[str, list[dict[str, float | int]]] = {}
        for (corpus_name, _filename), values in grouped.items():
            mean_probs = np.mean(values["gen_probs"], axis=0)
            per_corpus.setdefault(corpus_name, []).append(
                {
                    "gen_target": int(values["gen_target"]),
                    "gen_pred": int(np.argmax(mean_probs)),
                    "age_target": float(values["age_target"]),
                    "age_pred": float(np.mean(values["age_preds"])),
                }
            )

        return per_corpus

    def _compute_metrics(self, rows: list[dict[str, float | int]]) -> dict[str, float]:
        if not rows:
            return {}

        age_true = np.asarray([row["age_target"] for row in rows], dtype=np.float64)
        age_pred = np.asarray([row["age_pred"] for row in rows], dtype=np.float64)

        cm = self._compute_confusion_matrix(rows).astype(np.float64)

        tp = np.diag(cm)
        precision_den = cm.sum(axis=0)
        recall_den = cm.sum(axis=1)

        precision = np.divide(tp, precision_den, out=np.zeros_like(tp), where=precision_den != 0)
        recall = np.divide(tp, recall_den, out=np.zeros_like(tp), where=recall_den != 0)
        f1_den = precision + recall
        f1 = np.divide(2 * precision * recall, f1_den, out=np.zeros_like(tp), where=f1_den != 0)

        metrics = {
            "age_mae": float(np.mean(np.abs(age_pred - age_true))),
            "gen_precision": float(np.mean(precision)),
            "gen_uar": float(np.mean(recall)),
            "gen_macro_f1": float(np.mean(f1)),
            "num_files": float(len(rows)),
        }

        if len(rows) >= 2:
            age_true_centered = age_true - np.mean(age_true)
            age_pred_centered = age_pred - np.mean(age_pred)
            denom = np.sqrt(np.sum(age_true_centered**2)) * np.sqrt(np.sum(age_pred_centered**2))
            metrics["age_pcc"] = 0.0 if denom == 0.0 else float(
                np.sum(age_true_centered * age_pred_centered) / denom
            )

        return metrics

    def _compute_confusion_matrix(self, rows: list[dict[str, float | int]]) -> np.ndarray:
        y_true = np.asarray([row["gen_target"] for row in rows], dtype=np.int64)
        y_pred = np.asarray([row["gen_pred"] for row in rows], dtype=np.int64)
        return _confusion_matrix(y_true, y_pred, self.gender_num_classes)


@CALLBACKS.register("grouping_callback")
def grouping_callback(context = None, **params):
    gender_class_names = params.pop("gender_class_names", None)
    if gender_class_names is None and context is not None:
        gender_class_names = context.get("data.gender_class_names")
    
    return GroupingCallback(gender_class_names=gender_class_names, **params)
