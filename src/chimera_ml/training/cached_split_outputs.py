from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CachedSplitOutputs:
    """Cached per-sample model outputs for a given split.

    Stored on CPU to make it cheap to access from callbacks.
    """

    preds: torch.Tensor | list[torch.Tensor]
    targets: torch.Tensor | list[torch.Tensor] | None = None
    sample_meta: list[dict[str, Any]] | None = None
    features: torch.Tensor | list[torch.Tensor] | None = None

    @staticmethod
    def _concat_chunks(
        value: torch.Tensor | list[torch.Tensor] | None,
    ) -> torch.Tensor | None:
        """Concatenate cached chunk tensors into one CPU tensor when possible."""
        if value is None:
            return None

        if torch.is_tensor(value):
            return value.detach().cpu()

        if isinstance(value, list) and value:
            chunks = [v.detach().cpu() for v in value if torch.is_tensor(v)]
            if not chunks:
                return None
            try:
                return torch.cat(chunks, dim=0)
            except RuntimeError:
                return None

        return None
