from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chimera_ml.visualization.confusion_matrix import fig_to_png_bytes, plot_confusion_matrix


def test_plot_confusion_matrix_builds_figure_and_saves(tmp_path: Path):
    cm = np.array([[8, 2], [1, 9]], dtype=np.int64)
    out = tmp_path / "cm.png"

    fig = plot_confusion_matrix(
        cm,
        labels=["neg", "pos"],
        title="CM",
        save_path=str(out),
        colorbar=True,
        close=False,
    )

    assert fig is not None
    assert out.exists()
    assert fig.axes
    plt.close(fig)


def test_plot_confusion_matrix_validates_square_matrix():
    cm = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="square 2D array"):
        plot_confusion_matrix(cm)


def test_plot_confusion_matrix_validates_label_length():
    cm = np.array([[1, 0], [0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="labels length"):
        plot_confusion_matrix(cm, labels=["only_one"])


def test_plot_confusion_matrix_close_flag_closes_figure():
    cm = np.array([[1, 1], [0, 2]], dtype=np.int64)
    fig = plot_confusion_matrix(cm, close=True)
    assert not plt.fignum_exists(fig.number)


def test_fig_to_png_bytes_returns_png_signature():
    cm = np.array([[2, 0], [1, 3]], dtype=np.int64)
    fig = plot_confusion_matrix(cm)
    data = fig_to_png_bytes(fig)

    assert isinstance(data, bytes)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    plt.close(fig)
