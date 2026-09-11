from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class CheckpointCallback(BaseCallback):
    """Save the last and best model states during a training run.

    The callback writes checkpoints during on_epoch_end. It always saves
    last.pt when save_last is true, then evaluates monitor from the epoch log
    mapping and keeps the best checkpoint history according to mode. Missing
    monitor keys produce a warning and do not create a monitored checkpoint.

    Attributes:
        log_path: Root directory for experiment outputs.
        experiment_name: Experiment directory name below log_path.
        run_name: Run directory name below the experiment directory.
        monitor: Scalar log key to optimize, commonly val/loss.
        mode: min keeps lower values; max keeps higher values.
        save_top_k: Maximum number of monitored checkpoints to retain. A
            non-positive value disables pruning but the first best checkpoint
            is still written.
        save_last: Whether to write the rolling last.pt checkpoint.
        filename_template: Format string receiving epoch, step, monitor, and
            value for monitored checkpoint names.

    Checkpoint layout:
        Files are written to
        <log_path>/<experiment_name>/<run_name>/checkpoints.
        Payloads contain epoch, global_step, model_state_dict,
        optimizer_state_dict, and scheduler_state_dict when a scheduler is
        present.

    Note:
        The callback does not resume training or load checkpoints; it only
        serializes the current trainer state.
    """

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
        """Create the run checkpoint directory.

        Args:
            trainer: Active Trainer. Its state is not modified.
        """
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

        name = (
            "last.pt"
            if is_last
            else self.filename_template.format(
                epoch=epoch,
                step=step,
                monitor=self.monitor.replace("/", "_"),
                value=monitor_value,
            )
        )

        assert self._resolved_dirpath is not None
        path = self._resolved_dirpath / name
        torch.save(payload, path)
        return path

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        """Save last.pt and update the monitored checkpoint set.

        Args:
            trainer: Active Trainer providing model, optimizer, and optional
                scheduler state.
            epoch: One-based epoch number stored in the payload.
            logs: Aggregated scalar logs. monitor must be present to save a
                monitored checkpoint.
        """
        step = trainer.global_step

        if self.save_last:
            self._save(trainer, epoch=epoch, step=step, monitor_value=float("nan"), is_last=True)

        if self.monitor not in logs:
            available = ", ".join(sorted(logs.keys()))
            self._warning(
                trainer, f"[CheckpointCallback] monitor='{self.monitor}' not found in logs. Available keys: {available}"
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
