from abc import ABC, abstractmethod

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput


class BaseMetric(ABC):
    """Stateful metric extension contract.

    Trainer resets a metric at the start of an epoch, calls update for each
    labeled batch, and merges the mapping returned by compute into logs.
    Implementations should keep only epoch-local state.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated epoch state."""
        raise NotImplementedError

    @abstractmethod
    def update(self, output: ModelOutput, batch: Batch) -> None:
        """Accumulate predictions and targets from one batch.

        Args:
            output: Model predictions for the batch.
            batch: Inputs and labeled targets for the batch.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return metric names and values for the accumulated epoch."""
        raise NotImplementedError
