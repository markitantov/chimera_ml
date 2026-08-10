from dataclasses import dataclass, field
from typing import Any

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS
from chimera_ml.utils.sweep import SweepTarget


@dataclass
class SweepTargetCallback(BaseCallback):
    """Track the best scalar value for a sweep target."""

    monitor: str = "val/loss"
    mode: str = "min"

    best_value: float | None = field(init=False, default=None)
    best_epoch: int | None = field(init=False, default=None)
    last_value: float | None = field(init=False, default=None)
    last_epoch: int | None = field(init=False, default=None)
    available_keys: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        target = SweepTarget(monitor=self.monitor, mode=self.mode)
        self.monitor = target.monitor
        self.mode = target.mode

    def on_fit_start(self, trainer: Any) -> None:
        self.best_value = None
        self.best_epoch = None
        self.last_value = None
        self.last_epoch = None
        self.available_keys = ()

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        self.available_keys = tuple(sorted(logs.keys()))
        if self.monitor not in logs:
            return

        try:
            current = float(logs[self.monitor])
        except Exception as exc:
            raise ValueError(
                f"monitor '{self.monitor}' must be scalar-convertible, got {logs[self.monitor]!r}"
            ) from exc

        self.last_value = current
        self.last_epoch = epoch
        is_best = (
            self.best_value is None
            or (self.mode == "min" and current < self.best_value)
            or (self.mode == "max" and current > self.best_value)
        )

        if is_best:
            self.best_value = current
            self.best_epoch = epoch


@CALLBACKS.register("sweep_target_callback")
def sweep_target_callback(**params: Any) -> SweepTargetCallback:
    return SweepTargetCallback(**params)
