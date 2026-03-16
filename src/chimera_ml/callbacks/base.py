from abc import ABC
from typing import Any, Dict


class BaseCallback(ABC):
    """Callback hooks for training lifecycle."""

    def on_fit_start(self, trainer: Any) -> None:
        """On fit start.

        Args:
            trainer (Any): Trainer/engine instance coordinating the training loop.

        """
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """On epoch start.

        Args:
            trainer (Any): Additional parameter controlling the behavior of the method.
            epoch (int): Integer parameter.

        """
        pass

    def on_batch_end(self, trainer: Any, global_step: int, logs: Dict[str, float]) -> None:
        """On batch end.

        Args:
            trainer (Any): Additional parameter controlling the behavior of the method.
            global_step (int): Global optimization step counter.
            logs (Dict[str, float]): String parameter.

        """
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        """On epoch end.

        Args:
            trainer (Any): Additional parameter controlling the behavior of the method.
            epoch (int): Integer parameter.
            logs (Dict[str, float]): String parameter.

        """
        pass

    def on_fit_end(self, trainer: Any) -> None:
        """On fit end.

        Args:
            trainer (Any): Trainer/engine instance coordinating the training loop.

        """
        pass
