import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.base import BaseLoss
from chimera_ml.metrics.base import BaseMetric
from chimera_ml.training.config import TrainConfig
from chimera_ml.training.trainer import Trainer


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, batch: Batch) -> ModelOutput:
        preds = self.linear(batch.inputs["x"])
        return ModelOutput(preds=preds, aux={"features": preds + 1.0})


class _MSELoss(BaseLoss):
    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        return torch.mean((output.preds - batch.targets) ** 2)


class _Metric(BaseMetric):
    def reset(self) -> None:
        self._vals = []

    def update(self, output: ModelOutput, batch: Batch) -> None:
        self._vals.append(float(output.preds.detach().mean().item()))

    def compute(self) -> dict[str, float]:
        if not self._vals:
            return {}
        return {"mean_pred": float(sum(self._vals) / len(self._vals))}


class _SchedulerNeedsMetric:
    def __init__(self):
        self.calls = []

    def step(self, metrics, epoch=None):
        self.calls.append((metrics, epoch))


class _SchedulerNoMetric:
    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1


def _trainer() -> Trainer:
    model = _TinyModel()
    loss = _MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    cfg = TrainConfig(
        epochs=1,
        mixed_precision=False,
        use_scheduler=True,
        scheduler_monitor="val/loss",
        device="cpu",
    )
    return Trainer(
        model=model,
        loss_fn=loss,
        optimizer=opt,
        metrics=[_Metric()],
        config=cfg,
        mlflow_logger=None,
        logger=None,
        callbacks=[],
        scheduler=None,
    )


def _batch(has_targets: bool = True) -> Batch:
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0]], dtype=torch.float32) if has_targets else None
    return Batch(
        inputs={"x": x},
        targets=y,
        masks={"x_mask": torch.ones(2)},
        meta={"sample_meta": [{"id": "a"}, {"id": "b"}]},
    )


def test_trainer_move_to_device_and_to_device_helpers():
    tr = _trainer()
    cpu = torch.device("cpu")

    nested = {"a": torch.tensor([1.0]), "b": [torch.tensor([2.0]), (torch.tensor([3.0]), 4)]}
    moved = tr._move_to_device(nested, cpu)
    assert moved["a"].device.type == "cpu"
    assert moved["b"][0].device.type == "cpu"
    assert moved["b"][1][0].device.type == "cpu"

    out = tr._to_device(_batch(), cpu)
    assert out.inputs["x"].device.type == "cpu"
    assert out.targets is not None and out.targets.device.type == "cpu"


def test_trainer_extract_features_paths_and_errors():
    tr = _trainer()
    b = _batch()
    out = tr.model(b)

    feats = tr._extract_features(out, b, feature_extractor=None)
    assert feats.shape == out.preds.shape

    feats2 = tr._extract_features(out, b, feature_extractor=lambda *_: torch.zeros_like(out.preds))
    assert torch.allclose(feats2, torch.zeros_like(out.preds))

    try:
        tr._extract_features(out, b, feature_extractor=lambda *_: None)
    except ValueError as exc:
        assert "returned None" in str(exc)
    else:
        raise AssertionError("Expected ValueError for None features")

    try:
        tr._extract_features(ModelOutput(preds=out.preds, aux={}), b, feature_extractor=None)
    except ValueError as exc:
        assert "features were not found" in str(exc)
    else:
        raise AssertionError("Expected ValueError when aux features are missing")


def test_trainer_scheduler_helpers_cover_metric_and_non_metric():
    tr = _trainer()

    s_metric = _SchedulerNeedsMetric()
    s_nom = _SchedulerNoMetric()

    assert tr._scheduler_needs_metric(s_metric) is True
    assert tr._scheduler_needs_metric(s_nom) is False

    tr.scheduler = s_metric
    tr._scheduler_step({"val/loss": 0.2})
    assert s_metric.calls and float(s_metric.calls[0][0]) == 0.2

    tr._scheduler_step({"other": 1.0})
    assert len(s_metric.calls) == 1  # monitor not present

    tr.scheduler = s_nom
    tr._scheduler_step({"val/loss": 0.1})
    assert s_nom.calls == 1


def test_trainer_run_epoch_collects_cache_and_handles_target_errors():
    tr = _trainer()
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    metrics = tr._run_epoch(
        loader=[_batch()],
        device=torch.device("cpu"),
        scaler=scaler,
        train=False,
        epoch=1,
        split="val",
        with_features=True,
        feature_extractor=None,
    )

    assert "loss" in metrics
    assert "num_samples" in metrics
    cached = tr.get_cached_predictions("val")
    assert cached is not None
    assert cached.preds.shape[0] == 2
    assert cached.features is not None and cached.features.shape[0] == 2
    assert cached.sample_meta is not None and len(cached.sample_meta) == 2

    # Training requires targets
    try:
        tr._run_epoch(
            loader=[_batch(has_targets=False)],
            device=torch.device("cpu"),
            scaler=scaler,
            train=True,
            epoch=1,
            split="train",
        )
    except ValueError as exc:
        assert "has no targets" in str(exc)
    else:
        raise AssertionError("Expected ValueError for train batch without targets")


class _SeqModel(torch.nn.Module):
    def forward(self, batch: Batch) -> ModelOutput:
        # Keep sequence dimension from input, may vary between batches.
        preds = batch.inputs["x"]
        return ModelOutput(preds=preds, aux={"features": preds + 1.0})


def _seq_batch(t: int) -> Batch:
    x = torch.arange(0, 2 * t, dtype=torch.float32).view(2, t, 1)
    y = x + 0.5
    return Batch(
        inputs={"x": x},
        targets=y,
        masks=None,
        meta={"sample_meta": [{"id": f"{t}-0"}, {"id": f"{t}-1"}]},
    )


def test_trainer_run_epoch_keeps_ragged_cache_for_variable_sequence_length():
    model = _SeqModel()
    loss = _MSELoss()
    opt = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-3)
    cfg = TrainConfig(epochs=1, mixed_precision=False, use_scheduler=False, device="cpu")
    tr = Trainer(
        model=model,
        loss_fn=loss,
        optimizer=opt,
        metrics=[_Metric()],
        config=cfg,
        mlflow_logger=None,
        logger=None,
        callbacks=[],
        scheduler=None,
    )
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    tr._run_epoch(
        loader=[_seq_batch(3), _seq_batch(5)],
        device=torch.device("cpu"),
        scaler=scaler,
        train=False,
        epoch=1,
        split="val_seq",
        with_features=True,
    )

    cached = tr.get_cached_predictions("val_seq")
    assert cached is not None
    assert isinstance(cached.preds, list)
    assert isinstance(cached.targets, list)
    assert isinstance(cached.features, list)
    assert len(cached.preds) == 2
