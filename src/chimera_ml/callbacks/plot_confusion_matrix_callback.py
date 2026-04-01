import io
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import torch

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS
from chimera_ml.metrics._utils import compute_confusion_matrix
from chimera_ml.training.cached_split_outputs import CachedSplitOutputs


def _import_pyplot() -> Any:
    """Import matplotlib.pyplot lazily with actionable error."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Dependency 'matplotlib' is not installed. "
            "Install it with: pip install matplotlib"
        ) from e

    return plt


@dataclass
class PlotConfusionMatrixCallback(BaseCallback):
    """Render confusion matrix figures and log them as MLflow artifacts."""

    splits: list[str] = field(default_factory=lambda: ["val"])
    artifact_path: str = "figures"
    filename_template: str = "confusion_matrix_epoch_{epoch}.png"
    title_template: str = "{split} Confusion Matrix (epoch {epoch})"

    def on_fit_start(self, trainer: Any) -> None:
        """Ensure predictions are cached for requested splits."""
        trainer.config.collect_cache = True

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        """Render confusion matrix from cached predictions and upload to MLflow."""
        logger = getattr(trainer, "mlflow_logger", None)
        if logger is None:
            return

        plt = _import_pyplot()

        for split_name in self._resolve_splits(trainer):
            cached = self._get_cached_split_outputs(trainer, split_name)
            if cached is None or cached.targets is None:
                continue

            y_true, y_pred = self._extract_class_indices(cached)
            if y_true is None or y_pred is None:
                continue

            cm = compute_confusion_matrix(y_true=y_true, y_pred=y_pred)
            if cm.size == 0:
                continue

            fig = None
            try:
                fig = _plot_confusion_matrix(
                    cm=cm,
                    labels=getattr(trainer, "class_names", None),
                    title=self.title_template.format(split=split_name, epoch=epoch),
                )
                logger.log_artifact_bytes(
                    _fig_to_png_bytes(fig),
                    artifact_path=f"{self.artifact_path}/{split_name}",
                    filename=self.filename_template.format(split=split_name, epoch=epoch),
                )
            except Exception as exc:
                self._warning(
                    trainer,
                    f"[PlotConfusionMatrixCallback] Failed for split '{split_name}': {exc}",
                )
            finally:
                if fig is not None:
                    plt.close(fig)

    def _resolve_splits(self, trainer: Any) -> list[str]:
        """Resolve configured split selectors into concrete split names."""
        resolved: list[str] = []
        seen: set[str] = set()

        train_loaders = getattr(trainer, "_train_loaders", None)
        val_loaders = getattr(trainer, "_val_loaders", None)
        test_loaders = getattr(trainer, "_test_loaders", None)
        all_loaders = getattr(trainer, "_loaders", None)

        def add(name: str) -> None:
            if name and name not in seen:
                seen.add(name)
                resolved.append(name)

        for split in self.splits or ["val"]:
            if split == "train":
                if isinstance(train_loaders, dict) and train_loaders:
                    for name in train_loaders:
                        add(name)
                elif isinstance(all_loaders, dict) and "train" in all_loaders:
                    add("train")
                continue

            if split == "val":
                if isinstance(val_loaders, dict) and val_loaders:
                    for name in val_loaders:
                        add(name)
                elif isinstance(all_loaders, dict) and "val" in all_loaders:
                    add("val")
                continue

            if split == "test":
                if isinstance(test_loaders, dict) and test_loaders:
                    for name in test_loaders:
                        add(name)
                elif isinstance(all_loaders, dict) and "test" in all_loaders:
                    add("test")
                continue

            add(split)

        return resolved

    @staticmethod
    def _get_cached_split_outputs(trainer: Any, split: str) -> CachedSplitOutputs | None:
        getter = getattr(trainer, "get_cached_split_outputs", None)
        if not callable(getter):
            return None

        return getter(split)

    @classmethod
    def _extract_class_indices(
        cls, cached: CachedSplitOutputs
    ) -> tuple[Any | None, Any | None]:
        preds = CachedSplitOutputs._concat_chunks(cached.preds)
        targets = CachedSplitOutputs._concat_chunks(cached.targets)
        if preds is None or targets is None:
            return None, None

        y_pred = (
            preds.view(-1).to(torch.long)
            if preds.ndim <= 1
            else preds.argmax(dim=-1).view(-1).to(torch.long)
        )

        if targets.ndim > 1 and targets.shape[-1] > 1:
            y_true = targets.argmax(dim=-1).view(-1).to(torch.long)
        else:
            y_true = targets.view(-1).to(torch.long)

        n = int(min(y_true.numel(), y_pred.numel()))
        if n == 0:
            return None, None

        return y_true[:n].numpy(), y_pred[:n].numpy()


def _plot_confusion_matrix(
    cm: np.ndarray,
    *,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
    colorbar: bool = True,
    figsize: tuple[int, int] = (8, 6),
    color_map: Any | None = None,
    xticks_rotation: float | str = "horizontal",
    close: bool = False,
) -> Any:
    """Plot confusion matrix with absolute counts and row-normalized percentages."""
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be a square 2D array, got shape {cm.shape}")

    n_classes = cm.shape[0]

    if labels is None:
        labels = [str(i) for i in range(n_classes)]

    if len(labels) != n_classes:
        raise ValueError(f"labels length must be {n_classes}, got {len(labels)}")

    plt = _import_pyplot()
    cmap = color_map if color_map is not None else plt.cm.Blues

    cm = cm.astype(np.float64)

    row_sum = cm.sum(axis=1, keepdims=True)
    float_cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0) * 100.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(float_cm, interpolation="nearest", cmap=cmap)
    im.set_clim(0, 100)

    if colorbar:
        fig.colorbar(im, ax=ax)

    thresh = (float_cm.max() + float_cm.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        txt_color = "white" if float_cm[i, j] > thresh else "black"
        ax.text(
            j,
            i,
            f"{int(cm[i, j])}\n{float_cm[i, j]:.1f}%",
            ha="center",
            va="center",
            color=txt_color,
        )

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        title=title,
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if close:
        plt.close(fig)

    return fig


def _fig_to_png_bytes(fig: Any, dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


@CALLBACKS.register("plot_confusion_matrix_callback")
def plot_confusion_matrix_callback(**params):
    return PlotConfusionMatrixCallback(**params)
