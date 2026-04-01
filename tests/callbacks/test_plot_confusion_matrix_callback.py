import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chimera_ml.callbacks.plot_confusion_matrix_callback import (
    PlotConfusionMatrixCallback,
    _fig_to_png_bytes,
    _plot_confusion_matrix,
)
from chimera_ml.training.cached_split_outputs import CachedSplitOutputs

cm_plot_module = importlib.import_module("chimera_ml.callbacks.plot_confusion_matrix_callback")


class _MLflowStub:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str, str]] = []

    def log_artifact_bytes(self, data: bytes, artifact_path: str, filename: str) -> None:
        self.calls.append((data, artifact_path, filename))


class _LoggerStub:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args, **kwargs) -> None:
        if args:
            msg = msg % args
        self.warnings.append(str(msg))


class _TrainerStub:
    def __init__(self, *, with_mlflow: bool = True):
        self.config = SimpleNamespace(collect_cache=False)
        self.mlflow_logger = _MLflowStub() if with_mlflow else None
        self.logger = _LoggerStub()
        self._train_loaders = {}
        self._val_loaders = {"val": object()}
        self._test_loaders = {}
        self._loaders = {}
        self.cached_outputs = {
            "val": CachedSplitOutputs(
                preds=torch.tensor([[0.2, 0.8], [0.9, 0.1]], dtype=torch.float32),
                targets=torch.tensor([1, 0], dtype=torch.long),
            )
        }

    def get_cached_split_outputs(self, split: str) -> CachedSplitOutputs | None:
        return self.cached_outputs.get(split)


def test_plot_confusion_matrix_callback_enables_cache_collection():
    trainer = _TrainerStub()
    cb = PlotConfusionMatrixCallback(class_names=["neg", "pos"])
    cb.on_fit_start(trainer)
    assert trainer.config.collect_cache is True


def test_plot_confusion_matrix_callback_logs_artifact(monkeypatch):
    trainer = _TrainerStub()
    closed: list[object] = []

    class _PlotLib:
        @staticmethod
        def close(fig):
            closed.append(fig)

    def _plot(cm, labels, title):
        assert np.array_equal(cm, np.array([[1, 0], [0, 1]], dtype=np.int64))
        assert labels == ["neg", "pos"]
        assert title == "val Confusion Matrix (epoch 3)"
        return {"fig": "ok"}

    monkeypatch.setattr(cm_plot_module, "_import_pyplot", lambda: _PlotLib)
    monkeypatch.setattr(cm_plot_module, "_plot_confusion_matrix", _plot)
    monkeypatch.setattr(cm_plot_module, "_fig_to_png_bytes", lambda fig: b"png-bytes")

    cb = PlotConfusionMatrixCallback(class_names=["neg", "pos"])
    cb.on_epoch_end(trainer, epoch=3, logs={})

    assert trainer.mlflow_logger is not None
    assert trainer.mlflow_logger.calls == [(b"png-bytes", "figures/val", "confusion_matrix_epoch_3.png")]
    assert closed == [{"fig": "ok"}]


def test_plot_confusion_matrix_callback_uses_concat_chunks_for_cached_lists(monkeypatch):
    trainer = _TrainerStub()
    trainer.cached_outputs["val"] = CachedSplitOutputs(
        preds=[
            torch.tensor([[0.1, 0.9]], dtype=torch.float32),
            torch.tensor([[0.8, 0.2]], dtype=torch.float32),
        ],
        targets=[
            torch.tensor([1], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        ],
    )

    monkeypatch.setattr(cm_plot_module, "_import_pyplot", lambda: SimpleNamespace(close=lambda _: None))
    monkeypatch.setattr(cm_plot_module, "_fig_to_png_bytes", lambda fig: b"png-bytes")
    monkeypatch.setattr(cm_plot_module, "_plot_confusion_matrix", lambda **kwargs: {"fig": "ok"})

    cb = PlotConfusionMatrixCallback(class_names=["neg", "pos"])
    cb.on_epoch_end(trainer, epoch=2, logs={})

    assert trainer.mlflow_logger is not None
    assert trainer.mlflow_logger.calls == [(b"png-bytes", "figures/val", "confusion_matrix_epoch_2.png")]


def test_plot_confusion_matrix_callback_handles_ragged_sequence_chunks(monkeypatch):
    trainer = _TrainerStub()
    trainer.cached_outputs["val"] = CachedSplitOutputs(
        preds=[
            torch.tensor([[[0.1, 0.9], [0.8, 0.2]]], dtype=torch.float32),
            torch.tensor([[[0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]], dtype=torch.float32),
        ],
        targets=[
            torch.tensor([[1, 1]], dtype=torch.long),
            torch.tensor([[0, 1, 0]], dtype=torch.long),
        ],
    )

    class _PlotLib:
        @staticmethod
        def close(fig):
            return None

    captured: dict[str, object] = {}

    def _plot(cm, labels, title):
        captured["cm"] = cm
        captured["labels"] = labels
        captured["title"] = title
        return {"fig": "ok"}

    monkeypatch.setattr(cm_plot_module, "_import_pyplot", lambda: _PlotLib)
    monkeypatch.setattr(cm_plot_module, "_plot_confusion_matrix", _plot)
    monkeypatch.setattr(cm_plot_module, "_fig_to_png_bytes", lambda fig: b"png-bytes")

    cb = PlotConfusionMatrixCallback(class_names=["neg", "pos"])
    cb.on_epoch_end(trainer, epoch=5, logs={})

    assert trainer.mlflow_logger is not None
    assert trainer.mlflow_logger.calls == [(b"png-bytes", "figures/val", "confusion_matrix_epoch_5.png")]
    assert np.array_equal(captured["cm"], np.array([[1, 1], [1, 2]], dtype=np.int64))
    assert captured["labels"] == ["neg", "pos"]
    assert captured["title"] == "val Confusion Matrix (epoch 5)"


def test_plot_confusion_matrix_callback_supports_five_classes(monkeypatch):
    trainer = _TrainerStub()
    trainer.cached_outputs["val"] = CachedSplitOutputs(
        preds=torch.tensor(
            [
                [0.9, 0.1, 0.0, 0.0, 0.0],  # -> 0
                [0.0, 0.7, 0.2, 0.1, 0.0],  # -> 1
                [0.0, 0.1, 0.8, 0.1, 0.0],  # -> 2
                [0.0, 0.1, 0.1, 0.7, 0.1],  # -> 3
                [0.0, 0.0, 0.1, 0.2, 0.7],  # -> 4
            ],
            dtype=torch.float32,
        ),
        targets=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
    )

    class _PlotLib:
        @staticmethod
        def close(fig):
            return None

    captured: dict[str, object] = {}

    def _plot(cm, labels, title):
        captured["cm"] = cm
        captured["labels"] = labels
        captured["title"] = title
        return {"fig": "ok"}

    monkeypatch.setattr(cm_plot_module, "_import_pyplot", lambda: _PlotLib)
    monkeypatch.setattr(cm_plot_module, "_plot_confusion_matrix", _plot)
    monkeypatch.setattr(cm_plot_module, "_fig_to_png_bytes", lambda fig: b"png-bytes")

    cb = PlotConfusionMatrixCallback(class_names=["c0", "c1", "c2", "c3", "c4"])
    cb.on_epoch_end(trainer, epoch=6, logs={})

    assert trainer.mlflow_logger is not None
    assert trainer.mlflow_logger.calls == [(b"png-bytes", "figures/val", "confusion_matrix_epoch_6.png")]
    assert np.array_equal(captured["cm"], np.eye(5, dtype=np.int64))
    assert captured["labels"] == ["c0", "c1", "c2", "c3", "c4"]
    assert captured["title"] == "val Confusion Matrix (epoch 6)"


def test_plot_confusion_matrix_callback_supports_seven_classes_with_ragged_chunks(monkeypatch):
    trainer = _TrainerStub()

    def _logits_for_classes(classes: list[int]) -> torch.Tensor:
        logits = torch.full((len(classes), 7), -5.0, dtype=torch.float32)
        for i, cls in enumerate(classes):
            logits[i, cls] = 5.0
        return logits

    trainer.cached_outputs["val"] = CachedSplitOutputs(
        preds=[
            _logits_for_classes([0, 1, 2, 3]).unsqueeze(0),  # [1,4,7]
            _logits_for_classes([4, 5, 6]).unsqueeze(0),  # [1,3,7]
        ],
        targets=[
            torch.tensor([[0, 1, 2, 3]], dtype=torch.long),  # [1,4]
            torch.tensor([[4, 5, 6]], dtype=torch.long),  # [1,3]
        ],
    )

    class _PlotLib:
        @staticmethod
        def close(fig):
            return None

    captured: dict[str, object] = {}

    def _plot(cm, labels, title):
        captured["cm"] = cm
        captured["labels"] = labels
        captured["title"] = title
        return {"fig": "ok"}

    monkeypatch.setattr(cm_plot_module, "_import_pyplot", lambda: _PlotLib)
    monkeypatch.setattr(cm_plot_module, "_plot_confusion_matrix", _plot)
    monkeypatch.setattr(cm_plot_module, "_fig_to_png_bytes", lambda fig: b"png-bytes")

    cb = PlotConfusionMatrixCallback(class_names=[f"c{i}" for i in range(7)])
    cb.on_epoch_end(trainer, epoch=7, logs={})

    assert trainer.mlflow_logger is not None
    assert trainer.mlflow_logger.calls == [(b"png-bytes", "figures/val", "confusion_matrix_epoch_7.png")]
    assert np.array_equal(captured["cm"], np.eye(7, dtype=np.int64))
    assert captured["labels"] == [f"c{i}" for i in range(7)]
    assert captured["title"] == "val Confusion Matrix (epoch 7)"


def test_plot_confusion_matrix_callback_no_mlflow_is_noop(monkeypatch):
    trainer = _TrainerStub(with_mlflow=False)

    def _fail():
        raise AssertionError("plotting should not be called without mlflow logger")

    monkeypatch.setattr(cm_plot_module, "_import_pyplot", _fail)

    cb = PlotConfusionMatrixCallback()
    cb.on_epoch_end(trainer, epoch=1, logs={})


def test_plot_confusion_matrix_callback_requires_matplotlib(monkeypatch):
    trainer = _TrainerStub()

    monkeypatch.setattr(
        cm_plot_module,
        "_import_pyplot",
        lambda: (_ for _ in ()).throw(
            ModuleNotFoundError("Dependency 'matplotlib' is not installed.")
        ),
    )

    cb = PlotConfusionMatrixCallback()
    with pytest.raises(ModuleNotFoundError, match="matplotlib"):
        cb.on_epoch_end(trainer, epoch=1, logs={})


def test_plot_confusion_matrix_builds_figure_and_saves(tmp_path: Path):
    pytest.importorskip("matplotlib")
    cm = np.array([[8, 2], [1, 9]], dtype=np.int64)
    out = tmp_path / "cm.png"

    fig = _plot_confusion_matrix(
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
    cm_plot_module._import_pyplot().close(fig)


def test_plot_confusion_matrix_validates_square_matrix():
    cm = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="square 2D array"):
        _plot_confusion_matrix(cm)


def test_plot_confusion_matrix_validates_label_length():
    cm = np.array([[1, 0], [0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="labels length"):
        _plot_confusion_matrix(cm, labels=["only_one"])


def test_plot_confusion_matrix_close_flag_closes_figure():
    plt = pytest.importorskip("matplotlib.pyplot")
    cm = np.array([[1, 1], [0, 2]], dtype=np.int64)
    fig = _plot_confusion_matrix(cm, close=True)
    assert not plt.fignum_exists(fig.number)


def test_fig_to_png_bytes_returns_png_signature():
    pytest.importorskip("matplotlib")
    cm = np.array([[2, 0], [1, 3]], dtype=np.int64)
    fig = _plot_confusion_matrix(cm)
    data = _fig_to_png_bytes(fig)

    assert isinstance(data, bytes)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    cm_plot_module._import_pyplot().close(fig)
