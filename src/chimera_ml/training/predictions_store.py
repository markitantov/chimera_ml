from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EpochPredictions:
    """Cached per-sample predictions for a given split.

    Stored on CPU to make it cheap to access from callbacks.
    """

    preds: torch.Tensor | list[torch.Tensor]
    targets: torch.Tensor | list[torch.Tensor] | None = None
    sample_meta: list[dict[str, Any]] | None = None
    features: torch.Tensor | list[torch.Tensor] | None = None
