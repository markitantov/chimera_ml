from abc import ABC, abstractmethod

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput


class BaseLoss(ABC):
    """Loss contract used by Trainer.

    Implementations receive a ModelOutput and labeled Batch and return a
    differentiable tensor consumed by the optimizer.
    """

    @abstractmethod
    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        raise NotImplementedError
