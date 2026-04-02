from dataclasses import dataclass
from typing import Any

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
        self._best: float | None = None
        self._bad_epochs = 0
        self._should_stop = False
        self._countdown = self.patience

        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        if self.patience < 1:
            raise ValueError("patience must be >= 1")

    def _is_improvement(self, current: float, best: float) -> bool:
        """Return True if current metric improves beyond `min_delta`."""
        if self.mode == "min":
            return current < (best - self.min_delta)

        return current > (best + self.min_delta)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        """Update patience state and request early stop when needed."""
        if self.monitor not in logs:
            available = ", ".join(sorted(logs.keys()))
            self._warning(
                trainer,
                f"[EarlyStoppingCallback] monitor='{self.monitor}' not found in logs. Available keys: {available}",
            )

            return

        current = float(logs[self.monitor])
        if self._best is None:
            self._best = current
            self._bad_epochs = 0
            self._countdown = self.patience
            self._info(
                trainer,
                f"[EarlyStoppingCallback] epoch={epoch} {self.monitor}={current:.6f} "
                f"(best init). Countdown: {self._countdown}",
            )

            return

        if self._is_improvement(current, self._best):
            self._best = current
            self._bad_epochs = 0
            self._countdown = self.patience
            self._info(
                trainer,
                f"[EarlyStoppingCallback] epoch={epoch} {self.monitor}={current:.6f} "
                f"(improved). Reset countdown to {self._countdown}",
            )
        else:
            self._bad_epochs += 1
            self._countdown = max(0, self.patience - self._bad_epochs)

            self._info(
                trainer,
                f"[EarlyStoppingCallback] epoch={epoch} {self.monitor}={current:.6f} "
                f"(no improvement). Bad epochs: {self._bad_epochs}/{self.patience}. "
                f"Countdown: {self._countdown}",
            )

            if self._bad_epochs >= self.patience:
                self._should_stop = True
                trainer.stop_training = True
                self._info(
                    trainer,
                    f"[EarlyStoppingCallback] Stopping: no improvement in '{self.monitor}' "
                    f"for {self.patience} epochs. Best={self._best:.6f}, last={current:.6f}",
                )


@CALLBACKS.register("early_stopping_callback")
def early_stopping_callback(**params):
    """Registry factory for :class:`EarlyStoppingCallback`."""
    return EarlyStoppingCallback(**params)
