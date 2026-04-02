import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chimera_ml.core.registry import LOGGERS
from chimera_ml.logging.base import BaseLogger


def _import_mlflow() -> Any:
    """Import mlflow lazily and raise actionable error when dependency is missing."""
    try:
        import mlflow  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Dependency 'mlflow' is not installed. Install it with: pip install mlflow") from e
    return mlflow


@dataclass
class MLflowLogger(BaseLogger):
    tracking_uri: str | None = None
    experiment_name: str | None = "chimera"
    run_name: str | None = "train"
    config_path: str | None = None

    def __post_init__(self) -> None:
        """Initialize MLflow client and optional tracking URI/experiment."""
        self._mlflow = _import_mlflow()

        if self.tracking_uri:
            self._mlflow.set_tracking_uri(self.tracking_uri)

        if self.experiment_name:
            self._mlflow.set_experiment(self.experiment_name)

        self._active = False
        self._log_config = False

    def start(self, params: dict[str, Any] | None = None) -> None:
        """Start run and optionally log params and config artifact."""
        # Make it safe to call start() multiple times (CLI, Trainer, etc.)
        if not self._active:
            self._mlflow.start_run(run_name=self.run_name)
            self._active = True

        if params:
            self._mlflow.log_params(params)

        if not self._log_config and self.config_path:
            cfg = Path(self.config_path)
            if cfg.exists():
                self._mlflow.log_artifact(str(cfg.resolve()), artifact_path="configs")
                self._log_config = True

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        """Log an existing file as an MLflow artifact."""
        self._mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_artifact_bytes(self, data: bytes, artifact_path: str, filename: str) -> None:
        os.makedirs("/tmp/chimera_ml_artifacts", exist_ok=True)
        path = os.path.join("/tmp/chimera_ml_artifacts", filename)
        with open(path, "wb") as f:
            f.write(data)

        self._mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_text(self, text: str, artifact_path: str, filename: str) -> None:
        os.makedirs("/tmp/chimera_ml_artifacts", exist_ok=True)
        path = os.path.join("/tmp/chimera_ml_artifacts", filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        self._mlflow.log_artifact(path, artifact_path=artifact_path)

    def end(self) -> None:
        if self._active:
            self._mlflow.end_run()
            self._active = False


@LOGGERS.register("mlflow_logger")
def mlflow_logger(
    *,
    tracking_uri: str | None = None,
    experiment_name: str = "chimera",
    run_name: str = "train",
    config_path: str | None = None,
    **_,
) -> MLflowLogger:
    return MLflowLogger(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        config_path=config_path,
    )
