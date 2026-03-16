from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class CheckpointCallback(BaseCallback):
    """Save model checkpoints based on a monitored metric.
    """
    log_path: str = 'logs'
    experiment_name: str = 'chimera'
    run_name: str = 'train'
    monitor: str = "val/loss"
    mode: str = "min"  # "min" or "max"
    save_top_k: int = 1
    save_last: bool = True
    filename_template: str = "epoch={epoch}_step={step}_{monitor}={value:.4f}.pt"

    def __post_init__(self) -> None:
        self.best = None
        self.saved = []
        self._resolved_dirpath = None

        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

    def on_fit_start(self, trainer) -> None:
        # Resolve checkpoint directory once (per fit)
        dirpath = Path(self.log_path) / self.experiment_name / self.run_name / Path("checkpoints")
        dirpath.mkdir(parents=True, exist_ok=True)
        self._resolved_dirpath = dirpath

    def _is_better(self, current: float, best: float) -> bool:
        return current < best if self.mode == "min" else current > best

    def _sort_key(self, path: Path) -> float:
        return path.stat().st_mtime

    def _save(self, trainer, epoch: int, step: int, monitor_value: float, is_last: bool = False) -> Path:
        if self._resolved_dirpath is None:
            # Fallback: behave like before if callback wasn't wired into on_fit_start
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
        
        path = Path(self._resolved_dirpath) / name
        torch.save(payload, path)
        return path

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]) -> None:
        step = trainer.global_step

        if self.save_last:
            self._save(trainer, epoch=epoch, step=step, monitor_value=float("nan"), is_last=True)

        if self.monitor not in logs:
            available = ", ".join(sorted(logs.keys()))
            trainer.logger.warning(
                f"[EarlyStoppingCallback] monitor='{self.monitor}' not found in logs. "
                f"Available keys: {available}"
            )
            return

        current = float(logs[self.monitor])
        if self.best is None:
            self.best = current
            p = self._save(trainer, epoch, step, current, is_last=False)
            self.saved.append(p)
            return

        if self._is_better(current, self.best):
            self.best = current
            p = self._save(trainer, epoch, step, current, is_last=False)
            self.saved.append(p)

            if self.save_top_k > 0 and len(self.saved) > self.save_top_k:
                self.saved.sort(key=self._sort_key)
                while len(self.saved) > self.save_top_k:
                    to_remove = self.saved.pop(0)
                    if to_remove.exists() and to_remove.name != "last.pt":
                        to_remove.unlink(missing_ok=True)


@CALLBACKS.register("checkpoint_callback")
def checkpoint_callback(**params):
    return CheckpointCallback(**params)
