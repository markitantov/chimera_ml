from abc import ABC, abstractmethod

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput


class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        raise NotImplementedError
