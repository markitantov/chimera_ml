import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chimera_ml.callbacks.checkpoint_callback import CheckpointCallback
from chimera_ml.callbacks.collect_predictions_callback import CollectPredictionsCallback
from chimera_ml.callbacks.early_stopping_callback import EarlyStoppingCallback
from chimera_ml.callbacks.plot_confusion_matrix_callback import PlotConfusionMatrixCallback
from chimera_ml.callbacks.snapshot_callback import SnapshotCallback
from chimera_ml.callbacks.telegram_notifier_callback import TelegramNotifierCallback
from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.logging.base import BaseLogger
from chimera_ml.losses.base import BaseLoss
from chimera_ml.metrics.confusion_matrix_metric import ConfusionMatrixMetric
from chimera_ml.metrics.prf_metric import PRFMetric
from chimera_ml.training.config import TrainConfig
from chimera_ml.training.trainer import Trainer

plot_module = importlib.import_module("chimera_ml.callbacks.plot_confusion_matrix_callback")
telegram_module = importlib.import_module("chimera_ml.callbacks.telegram_notifier_callback")


class _ClassDataset(Dataset):
    def __init__(self, n: int, num_classes: int) -> None:
        self.n = int(n)
        self.num_classes = int(num_classes)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, object]:
        cls = int(idx % self.num_classes)
        x = torch.zeros(self.num_classes, dtype=torch.float32)
        x[cls] = 1.0
        return {
            "inputs": {"x": x},
            "target": torch.tensor(cls, dtype=torch.long),
            "meta": {"id": f"s{idx}"},
        }


def _collate(samples: list[dict[str, object]]) -> Batch:
    xs = torch.stack([sample["inputs"]["x"] for sample in samples], dim=0)
    ys = torch.stack([sample["target"] for sample in samples], dim=0)
    metas = [sample.get("meta", {}) for sample in samples]
    return Batch(
        inputs={"x": xs},
        targets=ys,
        masks={"x_mask": torch.ones(len(samples))},
        meta={"sample_meta": metas},
    )


class _LinearClassifier(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(num_classes, num_classes, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(num_classes, dtype=torch.float32))

    def forward(self, batch: Batch) -> ModelOutput:
        return ModelOutput(preds=self.fc(batch.inputs["x"]))


class _CrossEntropyLoss(BaseLoss):
    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        return F.cross_entropy(output.preds, batch.targets.long())


class _MemoryMLflowLogger(BaseLogger):
    experiment_name = "integration-exp"

    def __init__(self) -> None:
        self.started = 0
        self.ended = 0
        self.params: list[dict[str, object] | None] = []
        self.metric_calls: list[tuple[dict[str, float], int]] = []
        self.artifact_calls: list[tuple[str, str | None]] = []
        self.artifact_bytes_calls: list[tuple[bytes, str, str]] = []

    def start(self, params: dict[str, object] | None = None) -> None:
        self.started += 1
        self.params.append(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metric_calls.append((dict(metrics), int(step)))

    def end(self) -> None:
        self.ended += 1

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifact_calls.append((path, artifact_path))

    def log_artifact_bytes(self, data: bytes, artifact_path: str, filename: str) -> None:
        self.artifact_bytes_calls.append((data, artifact_path, filename))


class _ConsoleLoggerStub:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg: str, *args, **kwargs) -> None:
        if args:
            msg = msg % args
        self.infos.append(str(msg))

    def warning(self, msg: str, *args, **kwargs) -> None:
        if args:
            msg = msg % args
        self.warnings.append(str(msg))


class _ResponseStub:
    status_code = 200
    text = "ok"

    @staticmethod
    def raise_for_status() -> None:
        return None


class _TelegramSessionStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], int]] = []
        self.closed = False

    def post(self, url: str, data: dict[str, object], timeout: int) -> _ResponseStub:
        self.calls.append((url, data, timeout))
        return _ResponseStub()

    def close(self) -> None:
        self.closed = True


class _RequestsStub:
    RequestException = Exception

    def __init__(self, session: _TelegramSessionStub) -> None:
        self._session = session

    def Session(self) -> _TelegramSessionStub:
        return self._session


def test_trainer_fit_integrates_all_builtin_callbacks_and_accuracy_metrics(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")

    telegram_session = _TelegramSessionStub()
    monkeypatch.setattr(telegram_module, "_import_requests", lambda: _RequestsStub(telegram_session))

    plotted: dict[str, object] = {}
    monkeypatch.setattr(plot_module, "_import_pyplot", lambda: SimpleNamespace(close=lambda _fig: None))

    def _fake_plot(cm, labels, title):
        plotted["cm"] = cm
        plotted["labels"] = labels
        plotted["title"] = title
        return {"fig": "ok"}

    monkeypatch.setattr(plot_module, "_plot_confusion_matrix", _fake_plot)
    monkeypatch.setattr(plot_module, "_fig_to_png_bytes", lambda _fig: b"png-bytes")

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("seed: 0\n", encoding="utf-8")

    log_root = tmp_path / "logs"
    callbacks = [
        EarlyStoppingCallback(monitor="val/cm_acc", mode="max", patience=10),
        CheckpointCallback(
            log_path=str(log_root),
            experiment_name="exp",
            run_name="run",
            monitor="val/cm_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        SnapshotCallback(
            log_path=str(log_root),
            experiment_name="exp",
            run_name="run",
            include=["src/chimera_ml/callbacks/base.py"],
            save_code_zip=True,
            save_config=True,
            config_path=str(cfg_path),
        ),
        CollectPredictionsCallback(
            splits=["val"],
            task="classification",
            include_probs=True,
            artifact_path="predictions",
        ),
        PlotConfusionMatrixCallback(
            splits=["val"],
            class_names=["c0", "c1", "c2"],
            artifact_path="figures",
            title_template="{split} Confusion Matrix (epoch {epoch})",
        ),
        TelegramNotifierCallback(
            monitor="val/cm_acc",
            mode="max",
            include_trainer_progress=True,
            include_last_logs=True,
            include_best_logs=True,
            request_timeout_sec=5,
        ),
    ]

    model = _LinearClassifier(num_classes=3)
    trainer = Trainer(
        model=model,
        loss_fn=_CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        metrics=[
            ConfusionMatrixMetric(),
            PRFMetric(average="micro"),
            PRFMetric(average="macro"),
            PRFMetric(average="weighted"),
        ],
        config=TrainConfig(
            epochs=2,
            mixed_precision=False,
            use_scheduler=False,
            device="cpu",
            log_every_steps=1,
        ),
        mlflow_logger=_MemoryMLflowLogger(),
        logger=_ConsoleLoggerStub(),
        callbacks=callbacks,
        scheduler=None,
    )

    train_loader = DataLoader(_ClassDataset(n=12, num_classes=3), batch_size=4, collate_fn=_collate)
    val_loader = DataLoader(_ClassDataset(n=9, num_classes=3), batch_size=3, collate_fn=_collate)

    trainer.fit(train_loader, val_loaders={"val": val_loader})

    assert trainer.mlflow_logger is not None
    assert trainer.mlflow_logger.started == 1
    assert trainer.mlflow_logger.ended == 1

    metric_keys: set[str] = set()
    for metrics, _step in trainer.mlflow_logger.metric_calls:
        metric_keys.update(metrics.keys())

    assert "val/cm_acc" in metric_keys
    assert "val/micro_precision" in metric_keys
    assert "val/micro_recall" in metric_keys
    assert "val/micro_f1" in metric_keys
    assert "val/macro_precision" in metric_keys
    assert "val/macro_recall" in metric_keys
    assert "val/macro_f1" in metric_keys
    assert "val/weighted_precision" in metric_keys
    assert "val/weighted_recall" in metric_keys
    assert "val/weighted_f1" in metric_keys

    bytes_artifacts = trainer.mlflow_logger.artifact_bytes_calls
    assert any(path == "predictions/val" and name.startswith("preds_epoch_") for _b, path, name in bytes_artifacts)
    assert any(
        path == "figures/val" and name.startswith("confusion_matrix_epoch_")
        for _b, path, name in bytes_artifacts
    )

    artifact_files = trainer.mlflow_logger.artifact_calls
    assert any(path.endswith("code.zip") and artifact_path == "snapshots" for path, artifact_path in artifact_files)
    assert any(path.endswith("config.yaml") and artifact_path == "configs" for path, artifact_path in artifact_files)

    run_dir = log_root / "exp" / "run"
    ckpt_dir = run_dir / "checkpoints"
    assert (ckpt_dir / "last.pt").exists()
    assert len([p for p in ckpt_dir.glob("*.pt") if p.name != "last.pt"]) >= 1
    assert (run_dir / "code.zip").exists()
    assert (run_dir / "config.yaml").exists()

    cached = trainer.get_cached_split_outputs("val")
    assert cached is not None
    assert cached.targets is not None

    assert plotted
    assert np.asarray(plotted["cm"]).shape == (3, 3)
    assert plotted["labels"] == ["c0", "c1", "c2"]

    assert len(telegram_session.calls) == 1
    sent_text = str(telegram_session.calls[0][1]["text"])
    assert "Best value" in sent_text
    assert "val/cm_acc" in sent_text
    assert telegram_session.closed is True
