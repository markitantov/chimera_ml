from dataclasses import dataclass

import torch

from chimera_ml.callbacks.collect_predictions_callback import CollectPredictionsCallback


class _LoggerStub:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self.warnings.append(str(msg))


class _MLflowStub:
    def __init__(self):
        self.calls: list[tuple[bytes, str, str]] = []

    def log_artifact_bytes(self, data: bytes, artifact_path: str, filename: str):
        self.calls.append((data, artifact_path, filename))


@dataclass
class _Cached:
    preds: torch.Tensor
    targets: torch.Tensor | None = None
    sample_meta: list[dict[str, object]] | None = None


class _TrainerStub:
    def __init__(self):
        self.config = type("Cfg", (), {})()
        self.mlflow_logger = _MLflowStub()
        self.logger = _LoggerStub()
        self._train_loader = object()
        self._val_loader = object()
        self._test_loader = object()
        self._val_loaders = {"val_a": object(), "val_b": object()}
        self._test_loaders = {"test_a": object()}
        self._cache: dict[str, _Cached] = {}
        self.predict_calls: list[str] = []

    def get_cached_predictions(self, split: str):
        return self._cache.get(split)

    def predict(self, loader, split: str):
        self.predict_calls.append(split)
        return _Cached(
            preds=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            targets=torch.tensor([0, 1]),
            sample_meta=[{"id": "p0"}, {"id": "p1"}],
        )


def test_collect_predictions_sets_collect_cache_on_fit_start():
    trainer = _TrainerStub()
    cb = CollectPredictionsCallback()

    cb.on_fit_start(trainer)

    assert trainer.config.collect_cache is True


def test_collect_predictions_regression_uses_cache_and_logs_csv_bytes():
    trainer = _TrainerStub()
    trainer._cache["val_a"] = _Cached(
        preds=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        targets=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        sample_meta=[{"id": "a"}, {"id": "b"}],
    )
    cb = CollectPredictionsCallback(splits=["val_a"], task="regression")

    cb.on_epoch_end(trainer, epoch=2, logs={})

    assert len(trainer.mlflow_logger.calls) == 1
    data, artifact_path, filename = trainer.mlflow_logger.calls[0]
    text = data.decode("utf-8")
    assert "pred_0" in text and "target_1" in text and "id" in text
    assert artifact_path == "predictions/val_a"
    assert filename == "preds_epoch_2.csv"


def test_collect_predictions_classification_adds_probabilities_when_enabled():
    trainer = _TrainerStub()
    trainer._cache["test"] = _Cached(
        preds=torch.tensor([[2.0, 1.0], [0.1, 3.0]]),
        targets=torch.tensor([0, 1]),
        sample_meta=[{"id": "x"}, {"id": "y"}],
    )
    cb = CollectPredictionsCallback(splits=["test"], task="classification", include_probs=True)

    cb.on_epoch_end(trainer, epoch=1, logs={})

    data = trainer.mlflow_logger.calls[0][0].decode("utf-8")
    assert "pred_class" in data
    assert "prob_0" in data
    assert "prob_1" in data


def test_collect_predictions_falls_back_to_predict_when_cache_missing():
    trainer = _TrainerStub()
    cb = CollectPredictionsCallback(splits=["train"], task="classification")

    cb.on_epoch_end(trainer, epoch=3, logs={})

    assert trainer.predict_calls == ["train"]
    assert len(trainer.mlflow_logger.calls) == 1
    assert trainer.mlflow_logger.calls[0][1] == "predictions/train"


def test_collect_predictions_warns_when_predict_not_available_and_cache_missing():
    trainer = _TrainerStub()
    trainer.predict = None
    cb = CollectPredictionsCallback(splits=["train"], task="classification")

    cb.on_epoch_end(trainer, epoch=1, logs={})

    assert len(trainer.mlflow_logger.calls) == 0
    assert any("trainer.predict" in m for m in trainer.logger.warnings)


def test_collect_predictions_split_resolution_and_helpers_cover_edges():
    trainer = _TrainerStub()
    cb = CollectPredictionsCallback(splits=["val", "test", "train", "val_a", "missing"])

    resolved = cb._resolve_splits(trainer)
    names = [n for n, _ in resolved]
    assert "val_a" in names
    assert "val_b" in names
    assert "test_a" in names
    assert "train" in names
    assert "missing" in names

    ids = cb._extract_ids([{"id": "a"}, {"x": 1}, None], 3)
    assert ids == ["a", None, None]
    assert cb._extract_ids(None, 2) is None

    csv_bytes = cb._rows_to_csv_bytes([{"b": 2, "a": 1}])
    assert csv_bytes.decode("utf-8").startswith("a,b")


def test_collect_predictions_skips_when_no_mlflow_logger():
    trainer = _TrainerStub()
    trainer.mlflow_logger = None
    cb = CollectPredictionsCallback(splits=["val"])

    cb.on_epoch_end(trainer, epoch=1, logs={})

    assert trainer.predict_calls == []


def test_collect_predictions_handles_ragged_cached_tensors():
    trainer = _TrainerStub()
    trainer._cache["val_a"] = _Cached(
        preds=[
            torch.tensor([[[0.1], [0.2]], [[0.3], [0.4]]]),
            torch.tensor([[[0.5]], [[0.6]]]),
        ],
        targets=[
            torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
            torch.tensor([[[5.0]], [[6.0]]]),
        ],
        sample_meta=[{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
    )
    cb = CollectPredictionsCallback(splits=["val_a"], task="regression")

    cb.on_epoch_end(trainer, epoch=4, logs={})

    assert len(trainer.mlflow_logger.calls) == 1
    text = trainer.mlflow_logger.calls[0][0].decode("utf-8")
    assert "pred_0" in text
    assert "pred_1" in text
    assert "target_0" in text
