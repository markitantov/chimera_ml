from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch


def _pin_tensor_tree(x):
    if torch.is_tensor(x):
        return x.pin_memory()
    if isinstance(x, dict):
        return {k: _pin_tensor_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_pin_tensor_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_pin_tensor_tree(v) for v in x)
    return x


@dataclass
class Batch:
    # inputs[modality] -> tensor for that modality (may be missing in dict)
    inputs: Dict[str, torch.Tensor]
    # targets for task (classification/regression/etc.).
    # Can be None for unlabeled inference/test datasets.
    targets: Optional[torch.Tensor]
    # optional meta (e.g., ids, lengths, masks)
    meta: Optional[Dict[str, Any]] = None

    def get_masks(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return modality masks if present in meta."""
        if not self.meta:
            return None
        
        m = self.meta.get("masks")
        return m if isinstance(m, dict) else None

    # def pin_memory(self):
    #     self.inputs = {
    #         k: v.pin_memory() if torch.is_tensor(v) else v
    #         for k, v in self.inputs.items()
    #     }

    #     if self.targets is not None and torch.is_tensor(self.targets):
    #         self.targets = self.targets.pin_memory()

    #     return self
