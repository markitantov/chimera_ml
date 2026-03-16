from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class EpochPredictions:
    """Cached per-sample predictions for a given split.

    Stored on CPU to make it cheap to access from callbacks.
    """

    preds: torch.Tensor
    targets: Optional[torch.Tensor] = None
    sample_meta: Optional[List[Dict[str, Any]]] = None
    features: Optional[torch.Tensor] = None
