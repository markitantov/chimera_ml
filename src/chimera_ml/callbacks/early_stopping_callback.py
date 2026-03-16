from dataclasses import dataclass
from typing import Dict, Optional

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class EarlyStoppingCallback(BaseCallback):
    """Stop training when monitored metric stops improving."""

    monitor: str = "val/loss"
    mode: str = "min"  # "min" or "max"
    patience: int = 10
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        self.best = None
        self.bad_epochs = 0
        self.should_stop = False
        self.countdown = self.patience

        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]) -> None:
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
            self.bad_epochs = 0
            self.countdown = self.patience
            trainer.logger.info(
                f"[EarlyStoppingCallback] epoch={epoch} {self.monitor}={current:.6f} "
                f"(best init). Countdown: {self.countdown}"
            )
            return

        if self._is_improvement(current, self.best):
            self.best = current
            self.bad_epochs = 0
            self.countdown = self.patience
            trainer.logger.info(
                f"[EarlyStoppingCallback] epoch={epoch} {self.monitor}={current:.6f} "
                f"(improved). Reset countdown to {self.countdown}"
            )
        else:
            self.bad_epochs += 1
            self.countdown = max(0, self.patience - self.bad_epochs)

            trainer.logger.info(
                f"[EarlyStoppingCallback] epoch={epoch} {self.monitor}={current:.6f} "
                f"(no improvement). Bad epochs: {self.bad_epochs}/{self.patience}. "
                f"Countdown: {self.countdown}"
            )

            if self.bad_epochs >= self.patience:
                self.should_stop = True
                trainer.stop_training = True
                trainer.logger.info(
                    f"[EarlyStoppingCallback] Stopping: no improvement in '{self.monitor}' "
                    f"for {self.patience} epochs. Best={self.best:.6f}, last={current:.6f}"
                )


@CALLBACKS.register("early_stopping_callback")
def early_stopping_callback(**params):
    return EarlyStoppingCallback(**params)
