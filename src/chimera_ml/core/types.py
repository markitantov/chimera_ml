from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import torch


class Modality(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    PHYSIO = "physio"


@dataclass
class ModelOutput:
    # main predictions (logits for classification, values for regression)
    preds: torch.Tensor
    # optional extra tensors (embeddings, per-modality preds, etc.)
    aux: Optional[Dict[str, torch.Tensor]] = None
