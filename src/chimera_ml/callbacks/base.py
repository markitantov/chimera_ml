from typing import Any


class BaseCallback:
    """Extension contract for observing and extending the Trainer lifecycle.

    Callbacks are invoked in this order for a fit: on_fit_start; for every
    epoch, on_epoch_start, zero or more on_batch_end calls, and on_epoch_end;
    finally on_fit_end. The base implementation is a no-op.

    The individual lifecycle methods document their arguments and expected
    side effects.

    Note:
        Hooks may write artifacts or set trainer state, but should avoid
        mutating the model or optimizer outside the documented lifecycle.
    """

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
        """Prepare run-level state before a fit or evaluation loop.

        Args:
            trainer: Active Trainer whose state may be inspected or prepared.
        Side Effects:
            Implementations commonly create output directories or enable
            prediction caching. The base hook does nothing.
        """
        return

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Run before the first batch of an epoch.

        Args:
            trainer: Active Trainer instance.
            epoch: One-based epoch number.
        """
        return

    def on_batch_end(self, trainer: Any, global_step: int, logs: dict[str, float]) -> None:
        """Run after a training batch has completed.

        Args:
            trainer: Active Trainer instance.
            global_step: Current optimizer update count.
            logs: Scalar batch-level values available to the callback.
        Side Effects:
            Implementations may record progress or artifacts. The base hook
            does nothing.
        """
        return

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        """Run after train/validation processing for an epoch.

        Args:
            trainer: Active Trainer instance.
            epoch: One-based epoch number.
            logs: Aggregated train, validation, learning-rate, and other
                scalar values for the epoch.
        Side Effects:
            Implementations may save checkpoints, update stopping state, or
            publish metrics. The base hook does nothing.
        """
        return

    def on_fit_end(self, trainer: Any) -> None:
        """Finalize a fit or evaluation loop.

        Args:
            trainer: Active Trainer instance.
        Side Effects:
            Implementations may close sessions or send final notifications.
            The base hook does nothing.
        """
        return
