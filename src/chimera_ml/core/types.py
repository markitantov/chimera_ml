from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    """Standard model output container used by losses and metrics."""

    # main predictions (logits for classification, values for regression)
    preds: torch.Tensor
    # optional extra tensors (embeddings, per-modality preds, etc.)
    aux: dict[str, torch.Tensor] | None = None
