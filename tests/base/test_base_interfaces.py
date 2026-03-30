import pytest
import torch

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.logging.base import BaseLogger
from chimera_ml.losses.base import BaseLoss
from chimera_ml.metrics.base import BaseMetric
from chimera_ml.models.base import BaseModel


class _LoggerImpl(BaseLogger):
    def start(self, params=None):
        return None

    def log_metrics(self, metrics, step):
        return None

    def end(self):
        return None


class _MetricImpl(BaseMetric):
    def reset(self) -> None:
        return None

    def update(self, output: ModelOutput, batch: Batch) -> None:
        return None

    def compute(self) -> dict[str, float]:
        return {"x": 1.0}


class _LossImpl(BaseLoss):
    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        return torch.tensor(0.0)


class _ModelImpl(BaseModel):
    def forward(self, batch: Batch) -> ModelOutput:
        return ModelOutput(preds=torch.zeros(1, 1))


def test_base_logger_optional_methods_noop():
    logger = _LoggerImpl()
    logger.start(params={"a": 1})
    logger.log_metrics({"m": 1.0}, step=1)
    logger.log_artifact("x")
    logger.log_text("hello", "a", "b.txt")
    logger.log_artifact_bytes(b"x", "a", "b.bin")
    logger.end()


def test_abstract_base_classes_require_overrides():
    with pytest.raises(TypeError):
        BaseLogger()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        BaseMetric()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        BaseLoss()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


def test_minimal_base_implementations_work_and_callback_hooks_return_none(capsys):
    metric = _MetricImpl()
    loss = _LossImpl()
    model = _ModelImpl()

    batch = Batch(inputs={"x": torch.zeros(1, 1)}, targets=torch.zeros(1, 1))
    out = model(batch)

    assert loss(out, batch).ndim == 0
    assert metric.compute() == {"x": 1.0}

    cb = BaseCallback()
    assert cb.on_fit_start(object()) is None
    assert cb.on_epoch_start(object(), 1) is None
    assert cb.on_batch_end(object(), 1, {}) is None
    assert cb.on_epoch_end(object(), 1, {}) is None
    assert cb.on_fit_end(object()) is None

    BaseCallback._warning(type("T", (), {"logger": None})(), "hello")
    captured = capsys.readouterr()
    assert "hello" in captured.out
