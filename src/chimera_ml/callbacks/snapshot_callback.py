import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS
from chimera_ml.utils.utils import zip_sources


@dataclass
class SnapshotCallback(BaseCallback):
    """Save code.zip (from include) next to experiment checkpoints.
    """
    log_path: str = "logs"
    experiment_name: str = 'chimera'
    run_name: str = 'train'
    include: List[str] = field(default_factory=lambda: ["src"])
    save_code_zip: bool = True
    save_config: bool = True
    config_path: str = None

    def on_fit_start(self, trainer) -> None:
        dirpath = Path(self.log_path) / self.experiment_name / self.run_name
        dirpath.mkdir(parents=True, exist_ok=True)
        self._resolved_dirpath = dirpath

        # Code snapshot
        if self.save_code_zip:
            base = Path.cwd().resolve()
            zip_sources(zip_path=self._resolved_dirpath / "code.zip", base_dir=base, include=self.include)

        if self.save_config and self.config_path:
            shutil.copy2(self.config_path, self._resolved_dirpath / Path(self.config_path).name)

        # Log to MLflow
        if getattr(trainer, "mlflow_logger", None) is not None:
            try:
                trainer.mlflow_logger.log_artifacts(str(self._resolved_dirpath), artifact_path=self.include)
            except Exception:
                pass


@CALLBACKS.register("snapshot_callback")
def snapshot_callback(**params):
    return SnapshotCallback(**params)
