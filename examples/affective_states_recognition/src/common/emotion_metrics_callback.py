from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from common.utils import classification_metrics, cmu_multilabel_metrics

from chimera_ml.callbacks._utils import resolve_splits
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.callbacks.plot_confusion_matrix_callback import _fig_to_png_bytes, _plot_confusion_matrix
from chimera_ml.core.registry import CALLBACKS
from chimera_ml.metrics._utils import compute_confusion_matrix
from chimera_ml.training.cached_split_outputs import CachedSplitOutputs


@dataclass
class EmotionMetricsCallback(BaseCallback):
    splits: list[str] = field(default_factory=lambda: ["val"])
    emotion_class_names: list[str] | None = None
    sentiment_class_names: list[str] | None = None
    log_confusion_matrix: bool = True
    artifact_path: str = "figures/metrics"
    filename_template: str = "{task}_confusion_matrix_epoch_{epoch}.png"
    title_template: str = "{corpus} {task} Confusion Matrix (epoch {epoch})"

    def __post_init__(self) -> None:
        self.emotion_class_names = [
            str(x)
            for x in (self.emotion_class_names or ["neutral", "happy", "sad", "anger", "surprise", "disgust", "fear"])
        ]
        self.sentiment_class_names = [
            str(x) for x in (self.sentiment_class_names or ["negative", "neutral", "positive"])
        ]
        self.num_emotions = len(self.emotion_class_names)
        self.num_sentiments = len(self.sentiment_class_names)

    def on_fit_start(self, trainer: Any) -> None:
        trainer.config.collect_cache = True

    def find_loaders(self, trainer: Any) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        seen: set[str] = set()

        def add(name: str, loader: Any) -> None:
            if name and name not in seen:
                seen.add(name)
                out.append((name, loader))

        for split_name, loader in resolve_splits(trainer, self.splits):
            if loader is not None or trainer.get_cached_split_outputs(split_name) is not None:
                add(split_name, loader)

        selectors = [self.splits] if isinstance(self.splits, str) else list(self.splits) if self.splits else ["val"]
        for attr in ("_val_loaders", "_train_loaders", "_test_loaders", "_loaders"):
            loaders = getattr(trainer, attr, None)
            if not isinstance(loaders, dict):
                continue

            for selector in selectors:
                selector = str(selector)
                if selector in {"train", "val", "test"}:
                    for split_name, loader in loaders.items():
                        if split_name == selector or split_name.startswith(f"{selector}_"):
                            add(split_name, loader)

                elif selector in loaders:
                    add(selector, loaders[selector])

        return out

    @torch.no_grad()
    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        logger = getattr(trainer, "mlflow_logger", None)
        grouped: dict[tuple[str, str], dict[str, list[Any]]] = {}

        for split_name, _ in self.find_loaders(trainer):
            cached = trainer.get_cached_split_outputs(split_name)
            preds = CachedSplitOutputs._concat_chunks(cached.preds) if cached is not None else None
            targets = CachedSplitOutputs._concat_chunks(cached.targets) if cached is not None else None
            metas = list(cached.sample_meta or []) if cached is not None else []
            if preds is None or targets is None or not metas:
                continue

            phase_parts = str(split_name).replace("/", "_").split("_")
            phase = (
                phase_parts[1]
                if len(phase_parts) > 1
                and phase_parts[0] in {"train", "val", "test"}
                and phase_parts[1] in {"train", "dev", "devel", "val", "test"}
                else phase_parts[0]
            )
            phase = "devel" if phase in {"dev", "val", "devel"} else phase
            take = min(int(preds.shape[0]), int(targets.shape[0]), len(metas))
            emo_probs = torch.softmax(preds[:take, : self.num_emotions], dim=-1).cpu().numpy()
            sen_probs = (
                torch.softmax(preds[:take, self.num_emotions : self.num_emotions + self.num_sentiments], dim=-1)
                .cpu()
                .numpy()
            )

            emo_targets = targets[:take, : self.num_emotions].cpu().numpy()
            sen_targets = targets[:take, self.num_emotions : self.num_emotions + self.num_sentiments].cpu().numpy()

            for i, meta in enumerate(metas[:take]):
                corpus = str(meta.get("corpus_name", "UNKNOWN"))
                item = grouped.setdefault((phase, corpus), {"emo_t": [], "emo_p": [], "sen_t": [], "sen_p": []})
                if "CMUMOSEI" in corpus:
                    emo_t = (emo_targets[i] > 0).astype(np.int64)
                    emo_t = emo_t[1:] if emo_t.shape[0] > 6 else emo_t
                    neutral = emo_probs[i, 0] >= (1.0 - 1.0 / emo_probs.shape[1])
                    emo_p = np.zeros_like(emo_probs[i, 1:] if emo_probs.shape[1] > 6 else emo_probs[i], dtype=np.int64)
                    if emo_probs.shape[1] > 6 and not neutral:
                        emo_p = (emo_probs[i, 1:] >= (1.0 / emo_probs.shape[1])).astype(np.int64)
                    elif emo_probs.shape[1] <= 6:
                        emo_p = (emo_probs[i] >= 0.5).astype(np.int64)
                else:
                    emo_t = int(np.argmax(emo_targets[i]))
                    emo_p = int(np.argmax(emo_probs[i]))

                item["emo_t"].append(emo_t)
                item["emo_p"].append(emo_p)
                item["sen_t"].append(int(np.argmax(sen_targets[i])))
                item["sen_p"].append(int(np.argmax(sen_probs[i])))

        phase_scores: dict[str, list[float]] = {}
        phase_lines: dict[str, list[str]] = {}
        for (phase, corpus), item in sorted(grouped.items()):
            sen_t = np.asarray(item["sen_t"], dtype=np.int64)
            sen_p = np.asarray(item["sen_p"], dtype=np.int64)
            sen = classification_metrics(sen_t, sen_p)
            metrics = {
                "sen_A_WAR": sen["war"],
                "sen_UAR": sen["uar"],
                "sen_WF1": sen["weighted_f1"],
                "sen_MacroF1": sen["macro_f1"],
                "num_files": float(len(sen_t)),
            }

            if "CMUMOSEI" in corpus:
                emo_t = np.asarray(item["emo_t"], dtype=np.int64)
                emo_p = np.asarray(item["emo_p"], dtype=np.int64)
                emo, emo_pc = cmu_multilabel_metrics(emo_t, emo_p, class_names=self.emotion_class_names[1:7])
                metrics.update(emo)
                metrics["emo_sen_combined"] = (metrics["sen_MacroF1"] + metrics["emo_mMacroF1"]) / 2.0
            else:
                emo_t = np.asarray(item["emo_t"], dtype=np.int64)
                emo_p = np.asarray(item["emo_p"], dtype=np.int64)
                emo = classification_metrics(emo_t, emo_p)
                metrics.update(
                    {
                        "emo_A_WAR": emo["war"],
                        "emo_UAR": emo["uar"],
                        "emo_WF1": emo["weighted_f1"],
                        "emo_MacroF1": emo["macro_f1"],
                    }
                )
                metrics["emo_sen_combined"] = (metrics["sen_MacroF1"] + metrics["emo_MacroF1"]) / 2.0

            scope = f"{phase}_{corpus}_agg"
            scoped = {f"{scope}/{k}": float(v) for k, v in metrics.items()}
            logs.update(scoped)
            phase_scores.setdefault(phase, []).append(float(metrics["emo_sen_combined"]))
            lines = phase_lines.setdefault(phase, [])

            if "CMUMOSEI" in corpus:
                for agg, key in (
                    ("emo_mWA", "mWA"),
                    ("emo_mWF1", "mWF1"),
                    ("emo_mMacroF1", "mMacroF1"),
                    ("emo_mA_WAR", "mA_WAR"),
                ):
                    lines.append(
                        "[EmotionMetricsCallback] "
                        + f"{scope}/{agg}={metrics[agg]:.4f}"
                        + "".join(f" | {label}={value:.4f}" for label, value in emo_pc.get(key, {}).items())
                    )
            else:
                lines.append(
                    "[EmotionMetricsCallback] "
                    + " | ".join(
                        [
                            f"{scope}/emo_A_WAR={metrics['emo_A_WAR']:.4f}",
                            f"{scope}/emo_UAR={metrics['emo_UAR']:.4f}",
                            f"{scope}/emo_WF1={metrics['emo_WF1']:.4f}",
                            f"{scope}/emo_MacroF1={metrics['emo_MacroF1']:.4f}",
                        ]
                    )
                )
            lines.append(
                "[EmotionMetricsCallback] "
                + " | ".join(
                    [
                        f"{scope}/sen_A_WAR={metrics['sen_A_WAR']:.4f}",
                        f"{scope}/sen_UAR={metrics['sen_UAR']:.4f}",
                        f"{scope}/sen_WF1={metrics['sen_WF1']:.4f}",
                        f"{scope}/sen_MacroF1={metrics['sen_MacroF1']:.4f}",
                    ]
                )
            )
            lines.append(f"[EmotionMetricsCallback] {scope}/emo_sen_combined={metrics['emo_sen_combined']:.4f}")

            if logger is not None:
                logger.log_metrics(scoped, step=epoch)
                if self.log_confusion_matrix:
                    if "CMUMOSEI" not in corpus:
                        fig = _plot_confusion_matrix(
                            cm=compute_confusion_matrix(emo_t, emo_p),
                            labels=self.emotion_class_names,
                            title=self.title_template.format(corpus=corpus, task="Emotion", epoch=epoch),
                        )
                        logger.log_artifact_bytes(
                            _fig_to_png_bytes(fig),
                            artifact_path=f"{self.artifact_path}/{scope}",
                            filename=self.filename_template.format(task="emo", epoch=epoch),
                        )

                        plt.close(fig)
                    fig = _plot_confusion_matrix(
                        cm=compute_confusion_matrix(sen_t, sen_p),
                        labels=self.sentiment_class_names,
                        title=self.title_template.format(corpus=corpus, task="Sentiment", epoch=epoch),
                    )

                    logger.log_artifact_bytes(
                        _fig_to_png_bytes(fig),
                        artifact_path=f"{self.artifact_path}/{scope}",
                        filename=self.filename_template.format(task="sen", epoch=epoch),
                    )

                    plt.close(fig)

        for phase in ("train", "devel", "test"):
            for line in phase_lines.get(phase, []):
                self._info(trainer, line)

            values = phase_scores.get(phase, [])
            if not values:
                continue

            value = float(np.mean(values))
            logs[f"{phase}_all_agg/emo_sen_combined"] = value
            if logger is not None:
                logger.log_metrics({f"{phase}_all_agg/emo_sen_combined": value}, step=epoch)

            self._info(trainer, f"[EmotionMetricsCallback] {phase}_all_agg/emo_sen_combined={value:.4f}")


@CALLBACKS.register("emotion_metrics_callback")
def emotion_metrics_callback(context: Any | None = None, **params: Any) -> EmotionMetricsCallback:
    emotion_class_names = params.pop("emotion_class_names", None)
    if emotion_class_names is None and context is not None:
        emotion_class_names = context.get("data.emotion_class_names")

    sentiment_class_names = params.pop("sentiment_class_names", None)
    if sentiment_class_names is None and context is not None:
        sentiment_class_names = context.get("data.sentiment_class_names")

    return EmotionMetricsCallback(
        emotion_class_names=emotion_class_names,
        sentiment_class_names=sentiment_class_names,
        **params,
    )
