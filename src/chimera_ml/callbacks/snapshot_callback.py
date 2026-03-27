import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS
from chimera_ml.utils.utils import zip_sources


@dataclass
class SnapshotCallback(BaseCallback):
    """Save source/config snapshots and optionally log them as artifacts."""

    log_path: str = "logs"
    experiment_name: str = "chimera"
    run_name: str = "train"
    include: list[str] = field(default_factory=lambda: ["src"])
    save_code_zip: bool = True
    save_config: bool = True
    config_path: str | None = None

    def on_fit_start(self, trainer: Any) -> None:
        """Create snapshot artifacts and upload them to MLflow when available."""
        dirpath = Path(self.log_path) / self.experiment_name / self.run_name
        dirpath.mkdir(parents=True, exist_ok=True)
        self._resolved_dirpath = dirpath

        code_zip_path: Path | None = None
        cfg_copy_path: Path | None = None

        if self.save_code_zip:
            base = Path.cwd().resolve()
            code_zip_path = self._resolved_dirpath / "code.zip"
            zip_sources(zip_path=code_zip_path, base_dir=base, include=self.include)

        if self.save_config and self.config_path:
            cfg_copy_path = self._resolved_dirpath / Path(self.config_path).name
            shutil.copy2(self.config_path, cfg_copy_path)

        mlflow_logger = getattr(trainer, "mlflow_logger", None)
        if mlflow_logger is None:
            return

        try:
            if code_zip_path is not None and code_zip_path.exists():
                mlflow_logger.log_artifact(str(code_zip_path), artifact_path="snapshots")

            if cfg_copy_path is not None and cfg_copy_path.exists():
                mlflow_logger.log_artifact(str(cfg_copy_path), artifact_path="configs")
            
        except Exception as exc:
            self._warning(trainer, f"[SnapshotCallback] Failed to log snapshot artifacts: {exc}")


@CALLBACKS.register("snapshot_callback")
def snapshot_callback(**params):
    return SnapshotCallback(**params)
