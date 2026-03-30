import importlib
from pathlib import Path

from chimera_ml.core.registry import LOGGERS
from chimera_ml.logging.mlflow_logger import MLflowLogger, mlflow_logger


class _FakeMLflow:
    def __init__(self):
        self.tracking_uri = None
        self.experiment = None
        self.started = 0
        self.ended = 0
        self.logged_params = []
        self.logged_metrics = []
        self.logged_artifacts = []

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None):
        self.started += 1

    def end_run(self):
        self.ended += 1

    def log_params(self, params):
        self.logged_params.append(dict(params))

    def log_metrics(self, metrics, step=0):
        self.logged_metrics.append((dict(metrics), step))

    def log_artifact(self, path, artifact_path=None):
        self.logged_artifacts.append((str(path), artifact_path))


def test_mlflow_logger_start_log_and_end(monkeypatch, tmp_path: Path):
    fake = _FakeMLflow()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("seed: 1\n", encoding="utf-8")
    module = importlib.import_module("chimera_ml.logging.mlflow_logger")

    monkeypatch.setattr(module, "_import_mlflow", lambda: fake)

    logger = MLflowLogger(
        tracking_uri="sqlite:///mlruns.db",
        experiment_name="exp",
        run_name="run",
        config_path=str(cfg),
    )

    logger.start(params={"a": 1})
    logger.start(params={"b": 2})  # idempotent start_run, but params still logged

    logger.log_metrics({"loss": 0.1}, step=3)
    logger.log_artifact(str(cfg), artifact_path="cfg")
    logger.log_artifact_bytes(b"abc", artifact_path="bin", filename="x.bin")
    logger.log_text("hello", artifact_path="txt", filename="x.txt")

    logger.end()
    logger.end()  # idempotent end

    assert fake.tracking_uri == "sqlite:///mlruns.db"
    assert fake.experiment == "exp"
    assert fake.started == 1
    assert fake.ended == 1
    assert fake.logged_params == [{"a": 1}, {"b": 2}]
    assert fake.logged_metrics == [({"loss": 0.1}, 3)]

    # config + explicit artifact + bytes + text
    assert len(fake.logged_artifacts) == 4
    assert any(ap == "configs" for _, ap in fake.logged_artifacts)
    assert any(ap == "bin" for _, ap in fake.logged_artifacts)
    assert any(ap == "txt" for _, ap in fake.logged_artifacts)


def test_mlflow_factory_and_registry(monkeypatch):
    fake = _FakeMLflow()
    module = importlib.import_module("chimera_ml.logging.mlflow_logger")
    monkeypatch.setattr(module, "_import_mlflow", lambda: fake)

    factory = LOGGERS.get("mlflow_logger")
    assert callable(factory)

    logger = mlflow_logger(tracking_uri=None, experiment_name="exp", run_name="run", config_path=None)
    assert isinstance(logger, MLflowLogger)
