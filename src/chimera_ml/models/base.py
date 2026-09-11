from abc import ABC, abstractmethod

import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput


class BaseModel(nn.Module, ABC):
    """Abstract model contract for uni- and multimodal components.

    Implementations receive a Batch and return a ModelOutput. They may consume
    any subset of the modality keys present in Batch.inputs, but should raise a
    clear error when no usable input is available.
    """

    @abstractmethod
    def forward(self, batch: Batch) -> ModelOutput:
        raise NotImplementedError
