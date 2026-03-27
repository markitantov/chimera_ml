from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class CheckpointCallback(BaseCallback):
    """Save model checkpoints based on a monitored metric."""

    log_path: str = "logs"
    experiment_name: str = "chimera"
    run_name: str = "train"
    monitor: str = "val/loss"
    mode: str = "min"  # "min" or "max"
    save_top_k: int = 1
    save_last: bool = True
    filename_template: str = "epoch={epoch}_step={step}_{monitor}={value:.4f}.pt"

    def __post_init__(self) -> None:
        self._best: float | None = None
        self._saved: list[Path] = []
        self._resolved_dirpath: Path | None = None

        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

    def on_fit_start(self, trainer: Any) -> None:
        """Prepare checkpoint directory before the run starts."""
        dirpath = Path(self.log_path) / self.experiment_name / self.run_name / "checkpoints"
        dirpath.mkdir(parents=True, exist_ok=True)
        self._resolved_dirpath = dirpath

    def _is_better(self, current: float, best: float) -> bool:
        return current < best if self.mode == "min" else current > best

    def _sort_key(self, path: Path) -> float:
        return path.stat().st_mtime

    def _save(self, trainer: Any, epoch: int, step: int, monitor_value: float, is_last: bool = False) -> Path:
        """Serialize model/optimizer (and scheduler when available) to disk."""
        if self._resolved_dirpath is None:
            self.on_fit_start(trainer)

        payload = {
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        }
        
        if trainer.scheduler is not None:
            payload["scheduler_state_dict"] = trainer.scheduler.state_dict()

        name = "last.pt" if is_last else self.filename_template.format(
            epoch=epoch,
            step=step,
            monitor=self.monitor.replace("/", "_"),
            value=monitor_value,
        )

        assert self._resolved_dirpath is not None
        path = self._resolved_dirpath / name
        torch.save(payload, path)
        return path

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        """Optionally save last checkpoint and maintain top-k best checkpoints."""
        step = trainer.global_step

        if self.save_last:
            self._save(trainer, epoch=epoch, step=step, monitor_value=float("nan"), is_last=True)

        if self.monitor not in logs:
            available = ", ".join(sorted(logs.keys()))
            self._warning(
                trainer,
                f"[CheckpointCallback] monitor='{self.monitor}' not found in logs. "
                f"Available keys: {available}"
            )
            return

        current = float(logs[self.monitor])
        if self._best is None:
            self._best = current
            p = self._save(trainer, epoch, step, current, is_last=False)
            self._saved.append(p)
            return

        if self._is_better(current, self._best):
            self._best = current
            p = self._save(trainer, epoch, step, current, is_last=False)
            self._saved.append(p)

            if self.save_top_k > 0 and len(self._saved) > self.save_top_k:
                self._saved.sort(key=self._sort_key)
                while len(self._saved) > self.save_top_k:
                    to_remove = self._saved.pop(0)
                    if to_remove.exists() and to_remove.name != "last.pt":
                        to_remove.unlink(missing_ok=True)


@CALLBACKS.register("checkpoint_callback")
def checkpoint_callback(**params):
    """Registry factory for :class:`CheckpointCallback`."""
    return CheckpointCallback(**params)
