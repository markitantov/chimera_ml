from abc import ABC, abstractmethod

import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput


class BaseModel(nn.Module, ABC):
    """Base interface for uni- or multi-modal models."""

    @abstractmethod
    def forward(self, batch: Batch) -> ModelOutput:
        raise NotImplementedError
