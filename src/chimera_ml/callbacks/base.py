from typing import Any


class BaseCallback:
    """Base callback interface for the trainer lifecycle."""

    @classmethod
    def _log(cls, trainer: Any, level: str, message: str) -> None:
        """Get logger from trainer."""
        logger = getattr(trainer, "logger", None)
        if logger is not None and hasattr(logger, level):
            getattr(logger, level)(message)
            return

        print(message)

    @classmethod
    def _info(cls, trainer: Any, message: str) -> None:
        """Log informational message with fallback to stdout."""
        cls._log(trainer, "info", message)

    @classmethod
    def _warning(cls, trainer: Any, message: str) -> None:
        """Log warning message with fallback to stdout."""
        cls._log(trainer, "warning", message)

    @classmethod
    def _error(cls, trainer: Any, message: str) -> None:
        """Log error message with fallback to stdout."""
        cls._log(trainer, "error", message)

    def on_fit_start(self, trainer: Any) -> None:
        """Called once before the fit/eval loop starts."""
        return

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        return

    def on_batch_end(self, trainer: Any, global_step: int, logs: dict[str, float]) -> None:
        """Called after each training batch."""
        return

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        """Called after each epoch with aggregated logs."""
        return

    def on_fit_end(self, trainer: Any) -> None:
        """Called once after the fit/eval loop ends."""
        return
