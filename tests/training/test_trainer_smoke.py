import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import LOGGERS
from chimera_ml.core.types import ModelOutput
from chimera_ml.logging.base import BaseLogger
from chimera_ml.losses.base import BaseLoss
from chimera_ml.metrics.base import BaseMetric
from chimera_ml.training.builders import build_logger
from chimera_ml.training.config import TrainConfig
from chimera_ml.training.trainer import Trainer


class _TinyDataset(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.tensor([float(idx)], dtype=torch.float32)
        y = torch.tensor([float(idx)], dtype=torch.float32)
        return {"inputs": {"x": x}, "target": y}


def _collate(samples):
    xs = torch.stack([s["inputs"]["x"] for s in samples], dim=0)
    ys = torch.stack([s["target"] for s in samples], dim=0)
    return Batch(inputs={"x": xs}, targets=ys, masks={"x_mask": torch.ones(len(samples))}, meta={})


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, batch: Batch) -> ModelOutput:
        return ModelOutput(preds=self.linear(batch.inputs["x"]))


class _MSELoss(BaseLoss):
    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        return torch.mean((output.preds - batch.targets) ** 2)


class _MeanPredMetric(BaseMetric):
    def reset(self) -> None:
        self._vals = []

    def update(self, output: ModelOutput, batch: Batch) -> None:
        self._vals.append(float(output.preds.detach().mean().item()))

    def compute(self):
        if not self._vals:
            return {}
        return {"mean_pred": float(sum(self._vals) / len(self._vals))}


class _MemoryMLflowLogger(BaseLogger):
    def __init__(self) -> None:
        self.started = 0
        self.ended = 0
        self.logged: list[tuple[dict[str, float], int]] = []

    def start(self, params: dict[str, object] | None = None) -> None:
        self.started += 1

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.logged.append((dict(metrics), int(step)))

    def end(self) -> None:
        self.ended += 1


class _MemoryConsoleLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, msg: str, *args, **kwargs) -> None:
        if args:
            msg = msg % args
        self.messages.append(str(msg))


def _make_loader(n: int, batch_size: int = 2):
    return DataLoader(_TinyDataset(n), batch_size=batch_size, shuffle=False, collate_fn=_collate)


def _make_trainer(
    *,
    mlflow_logger: BaseLogger | None,
    logger: object | None,
) -> Trainer:
    model = _TinyModel()
    loss_fn = _MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    metric = _MeanPredMetric()
    cfg = TrainConfig(
        epochs=1,
        mixed_precision=False,
        use_scheduler=False,
        device="cpu",
        log_every_steps=1,
    )
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=[metric],
        config=cfg,
        mlflow_logger=mlflow_logger,
        logger=logger,
        callbacks=[],
        scheduler=None,
    )


def test_trainer_fit_smoke_runs_one_epoch():
    trainer = _make_trainer(mlflow_logger=None, logger=None)

    train_loader = _make_loader(4, batch_size=2)
    val_loader = _make_loader(2, batch_size=2)
    trainer.fit(train_loader, val_loaders={"val": val_loader})

    assert trainer.global_step > 0


def test_trainer_evaluate_returns_prefixed_metrics():
    model = _TinyModel()
    loss_fn = _MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    metric = _MeanPredMetric()
    cfg = TrainConfig(epochs=1, mixed_precision=False, use_scheduler=False, device="cpu")

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=[metric],
        config=cfg,
        mlflow_logger=None,
        logger=None,
        callbacks=[],
        scheduler=None,
    )

    out = trainer.evaluate({"val": _make_loader(3, batch_size=2)})
    assert "val/loss" in out
    assert "val/mean_pred" in out


@pytest.mark.parametrize("which_present", ["only_mlflow", "only_console"])
def test_trainer_fit_smoke_runs_with_partial_loggers(which_present: str):
    mlflow_logger: BaseLogger | None = (
        _MemoryMLflowLogger() if which_present == "only_mlflow" else None
    )
    console_logger = _MemoryConsoleLogger() if which_present == "only_console" else None
    trainer = _make_trainer(mlflow_logger=mlflow_logger, logger=console_logger)

    trainer.fit(_make_loader(4, batch_size=2), val_loaders={"val": _make_loader(2, batch_size=2)})

    assert trainer.global_step > 0
    if isinstance(mlflow_logger, _MemoryMLflowLogger):
        assert mlflow_logger.started == 1
        assert mlflow_logger.ended == 1
        assert len(mlflow_logger.logged) > 0
    if isinstance(console_logger, _MemoryConsoleLogger):
        assert len(console_logger.messages) > 0


def _register_test_custom_logger() -> str:
    key = "test_custom_memory_logger"
    registry_keys = LOGGERS.keys()
    if key not in registry_keys:
        @LOGGERS.register(key)
        def _factory(**_):
            return _MemoryMLflowLogger()

    return key


def test_trainer_fit_smoke_runs_with_registered_custom_logger():
    logger_key = _register_test_custom_logger()
    custom_logger = build_logger({"name": logger_key, "params": {}})
    assert isinstance(custom_logger, _MemoryMLflowLogger)

    trainer = _make_trainer(mlflow_logger=custom_logger, logger=None)
    trainer.fit(_make_loader(4, batch_size=2), val_loaders={"val": _make_loader(2, batch_size=2)})

    assert trainer.global_step > 0
    assert custom_logger.started == 1
    assert custom_logger.ended == 1
    assert len(custom_logger.logged) > 0
