from abc import ABC, abstractmethod

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput


class BaseMetric(ABC):
    """Stateful metric interface."""

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, output: ModelOutput, batch: Batch) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> dict[str, float]:
        raise NotImplementedError
