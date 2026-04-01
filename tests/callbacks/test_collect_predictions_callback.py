from dataclasses import dataclass

import torch

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.callbacks.collect_predictions_callback import CollectPredictionsCallback
from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss
from chimera_ml.metrics.regression_metric import MAEMetric
from chimera_ml.training.config import TrainConfig
from chimera_ml.training.trainer import Trainer


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
        self._train_loaders = {"train": object()}
        self._val_loaders = {"val_a": object(), "val_b": object()}
        self._test_loaders = {"test_a": object()}
        self._loaders = {}
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


class _SeqModel(torch.nn.Module):
    def forward(self, batch: Batch) -> ModelOutput:
        return ModelOutput(preds=batch.inputs["x"])


class _MSELoss(BaseLoss):
    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        return torch.mean((output.preds - batch.targets) ** 2)


def _seq_batch(t: int) -> Batch:
    x = torch.arange(0, 2 * t, dtype=torch.float32).view(2, t, 1)
    y = x + 0.5
    return Batch(
        inputs={"x": x},
        targets=y,
        masks={"x_mask": torch.ones(2)},
        meta={"sample_meta": [{"id": f"{t}-0"}, {"id": f"{t}-1"}]},
    )


def _seq_batch_equal_target(t: int) -> Batch:
    x = torch.arange(0, 2 * t, dtype=torch.float32).view(2, t, 1)
    return Batch(
        inputs={"x": x},
        targets=x.clone(),
        masks={"x_mask": torch.ones(2)},
        meta={"sample_meta": [{"id": f"eq-{t}-0"}, {"id": f"eq-{t}-1"}]},
    )


class _CacheMAEOnValCallback(BaseCallback):
    def __init__(self, split: str = "val") -> None:
        self.split = split
        self.last_mae: float | None = None

    @staticmethod
    def _as_chunks(x: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        if torch.is_tensor(x):
            return [x]
        return list(x)

    def on_epoch_end(self, trainer, epoch: int, logs: dict[str, float]) -> None:
        cached = trainer.get_cached_predictions(self.split)
        if cached is None or cached.targets is None:
            raise AssertionError(f"No cached preds/targets for split '{self.split}'.")

        preds_chunks = self._as_chunks(cached.preds)
        target_chunks = self._as_chunks(cached.targets)
        if len(preds_chunks) != len(target_chunks):
            raise AssertionError("Cached preds/targets chunks mismatch.")

        metric = MAEMetric()
        metric.reset()
        for pred_chunk, target_chunk in zip(preds_chunks, target_chunks, strict=True):
            # Flatten each chunk to keep shape-compatible updates for ragged sequence lengths.
            out = ModelOutput(preds=pred_chunk.reshape(-1))
            batch = Batch(inputs={}, targets=target_chunk.reshape(-1))
            metric.update(out, batch)

        values = metric.compute()
        mae = float(values["mae"])
        self.last_mae = mae
        logs[f"{self.split}/cache_mae"] = mae


def test_collect_predictions_integration_uses_trainer_ragged_cache():
    model = _SeqModel()
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-3)
    trainer = Trainer(
        model=model,
        loss_fn=_MSELoss(),
        optimizer=optimizer,
        metrics=[],
        config=TrainConfig(epochs=1, mixed_precision=False, use_scheduler=False, device="cpu"),
        mlflow_logger=_MLflowStub(),
        logger=None,
        callbacks=[],
        scheduler=None,
    )
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    trainer._run_epoch(
        loader=[_seq_batch(3), _seq_batch(5)],
        device=torch.device("cpu"),
        scaler=scaler,
        train=False,
        epoch=1,
        split="val_seq",
    )

    cached = trainer.get_cached_predictions("val_seq")
    assert cached is not None
    assert isinstance(cached.preds, list)
    assert isinstance(cached.targets, list)

    callback = CollectPredictionsCallback(splits=["val_seq"], task="regression")
    callback.on_epoch_end(trainer, epoch=7, logs={})

    assert trainer.mlflow_logger is not None
    assert len(trainer.mlflow_logger.calls) == 1
    data, artifact_path, filename = trainer.mlflow_logger.calls[0]
    text = data.decode("utf-8")
    assert artifact_path == "predictions/val_seq"
    assert filename == "preds_epoch_7.csv"
    assert "id" in text
    assert "pred_4" in text
    assert "target_4" in text


def test_validation_cache_callback_computes_mae_from_ragged_predictions():
    model = _SeqModel()
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-3)
    trainer = Trainer(
        model=model,
        loss_fn=_MSELoss(),
        optimizer=optimizer,
        metrics=[],
        config=TrainConfig(epochs=1, mixed_precision=False, use_scheduler=False, device="cpu"),
        mlflow_logger=None,
        logger=None,
        callbacks=[],
        scheduler=None,
    )
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    trainer._run_epoch(
        loader=[_seq_batch_equal_target(3), _seq_batch_equal_target(5)],
        device=torch.device("cpu"),
        scaler=scaler,
        train=False,
        epoch=1,
        split="val_seq",
    )

    callback = _CacheMAEOnValCallback(split="val_seq")
    logs: dict[str, float] = {}
    callback.on_epoch_end(trainer, epoch=1, logs=logs)

    assert callback.last_mae is not None
    assert callback.last_mae == 0.0
    assert "val_seq/cache_mae" in logs
