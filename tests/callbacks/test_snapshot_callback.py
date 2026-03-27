import importlib
from pathlib import Path

from chimera_ml.callbacks.snapshot_callback import SnapshotCallback

snapshot_module = importlib.import_module("chimera_ml.callbacks.snapshot_callback")


class _MlflowLoggerStub:
    def __init__(self):
        self.calls = []

    def log_artifact(self, path, artifact_path=None):
        self.calls.append((path, artifact_path))


class _TrainerStub:
    def __init__(self, with_mlflow: bool, with_logger: bool = False):
        self.mlflow_logger = _MlflowLoggerStub() if with_mlflow else None
        self.logger = None if not with_logger else _LoggerStub()


class _LoggerStub:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self.warnings.append(str(msg))


class _MlflowLoggerFailStub:
    def log_artifact(self, path, artifact_path=None):
        raise RuntimeError("mlflow is unavailable")


def test_snapshot_callback_copies_config_and_calls_zip(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("seed: 1\n", encoding="utf-8")

    called = {"zip": False}

    def _zip_stub(zip_path, base_dir, include):
        called["zip"] = True
        Path(zip_path).write_bytes(b"zip")

    monkeypatch.setattr(snapshot_module, "zip_sources", _zip_stub)

    cb = SnapshotCallback(
        log_path=str(tmp_path),
        experiment_name="exp",
        run_name="run",
        include=["src"],
        save_code_zip=True,
        save_config=True,
        config_path=str(cfg),
    )
    cb.on_fit_start(_TrainerStub(with_mlflow=False))

    out_dir = tmp_path / "exp" / "run"
    assert called["zip"] is True
    assert (out_dir / "code.zip").exists()
    assert (out_dir / "cfg.yaml").exists()


def test_snapshot_callback_handles_missing_log_artifacts_gracefully(tmp_path: Path):
    cb = SnapshotCallback(
        log_path=str(tmp_path),
        experiment_name="exp",
        run_name="run",
        save_code_zip=False,
        save_config=False,
    )
    cb.on_fit_start(_TrainerStub(with_mlflow=True))
    assert (tmp_path / "exp" / "run").exists()


def test_snapshot_callback_logs_artifacts_via_log_artifact(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("seed: 1\n", encoding="utf-8")

    trainer = _TrainerStub(with_mlflow=True)
    cb = SnapshotCallback(
        log_path=str(tmp_path),
        experiment_name="exp",
        run_name="run",
        include=["src"],
        save_code_zip=False,
        save_config=True,
        config_path=str(cfg),
    )
    cb.on_fit_start(trainer)

    assert len(trainer.mlflow_logger.calls) == 1
    path, artifact_path = trainer.mlflow_logger.calls[0]
    assert path.endswith("cfg.yaml")
    assert artifact_path == "configs"


def test_snapshot_callback_logs_code_zip_artifact(monkeypatch, tmp_path: Path):
    called = {"zip": False}

    def _zip_stub(zip_path, base_dir, include):
        called["zip"] = True
        Path(zip_path).write_bytes(b"zip")

    monkeypatch.setattr(snapshot_module, "zip_sources", _zip_stub)

    trainer = _TrainerStub(with_mlflow=True)
    cb = SnapshotCallback(
        log_path=str(tmp_path),
        experiment_name="exp",
        run_name="run",
        include=["src"],
        save_code_zip=True,
        save_config=False,
    )
    cb.on_fit_start(trainer)

    assert called["zip"] is True
    assert len(trainer.mlflow_logger.calls) == 1
    path, artifact_path = trainer.mlflow_logger.calls[0]
    assert path.endswith("code.zip")
    assert artifact_path == "snapshots"


def test_snapshot_callback_mlflow_failure_logs_warning(monkeypatch, tmp_path: Path):
    called = {"zip": False}

    def _zip_stub(zip_path, base_dir, include):
        called["zip"] = True
        Path(zip_path).write_bytes(b"zip")

    monkeypatch.setattr(snapshot_module, "zip_sources", _zip_stub)

    trainer = _TrainerStub(with_mlflow=False, with_logger=True)
    trainer.mlflow_logger = _MlflowLoggerFailStub()
    cb = SnapshotCallback(
        log_path=str(tmp_path),
        experiment_name="exp",
        run_name="run",
        include=["src"],
        save_code_zip=True,
        save_config=False,
    )
    cb.on_fit_start(trainer)

    assert called["zip"] is True
    assert any("Failed to log snapshot artifacts" in msg for msg in trainer.logger.warnings)
