import io
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
    colorbar: bool = True,
    figsize: tuple[int, int] = (8, 6),
    color_map: LinearSegmentedColormap = plt.cm.Blues,
    xticks_rotation: float | str = "horizontal",
    close: bool = False,
) -> Figure:
    """Plot confusion matrix with absolute counts and row-normalized percentages.

    Args:
        cm: Confusion matrix values of shape (C, C).
        labels: Class names (length C). If None, uses 0..C-1.
        title: Plot title.
        save_path: If provided, saves figure to this path.
        colorbar: Whether to draw a colorbar.
        figsize: Figure size.
        color_map: Matplotlib colormap.
        xticks_rotation: X tick label rotation.
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be a square 2D array, got shape {cm.shape}")

    n_classes = cm.shape[0]

    if labels is None:
        labels = [str(i) for i in range(n_classes)]
    if len(labels) != n_classes:
        raise ValueError(f"labels length must be {n_classes}, got {len(labels)}")

    cm = cm.astype(np.float64)

    # Row-normalized percentages (per true class)
    row_sum = cm.sum(axis=1, keepdims=True)
    float_cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0) * 100.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(float_cm, interpolation="nearest", cmap=color_map)
    im.set_clim(0, 100)

    if colorbar:
        fig.colorbar(im, ax=ax)

    # Choose text color based on background intensity
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

    # Fix matplotlib y-limits for imshow + text alignment
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

    fig.tight_layout()

    if save_path:
        # If format can't be inferred from extension, matplotlib falls back to default.
        plt.savefig(save_path)

    if close:
        plt.close(fig)

    return fig


def fig_to_png_bytes(fig: Figure, dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()
