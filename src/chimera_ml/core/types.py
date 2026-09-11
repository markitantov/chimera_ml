from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    """Standard model result consumed by losses, metrics, and callbacks.

    Attributes:
        preds: Main predictions, such as class logits or regression values.
        aux: Optional named tensors such as embeddings, modality predictions,
            or extracted features.
    """

    # main predictions (logits for classification, values for regression)
    preds: torch.Tensor
    # optional extra tensors (embeddings, per-modality preds, etc.)
    aux: dict[str, torch.Tensor] | None = None
