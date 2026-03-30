from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Batch:
    """Typed container for model inputs, targets, masks, and metadata."""

    # inputs[modality] -> tensor for that modality (may be missing in dict)
    inputs: dict[str, torch.Tensor]
    # targets for task (classification/regression/etc.).
    # Can be None for unlabeled inference/test datasets.
    targets: torch.Tensor | None
    # canonical container for masks (flat dict), e.g.:
    # {"sequence_mask": ..., "audio_mask": ..., "video_mask": ...}
    masks: dict[str, Any] | None = None
    # optional meta (e.g., ids, lengths, aux info)
    meta: dict[str, Any] | None = None

    @staticmethod
    def _flatten_legacy_masks(masks: dict[str, Any]) -> dict[str, Any]:
        """Flatten legacy nested masks from {'mask': {'audio': ...}} to {'audio_mask': ...}."""
        legacy_nested = masks.get("mask")
        if not isinstance(legacy_nested, dict):
            return masks

        flattened: dict[str, Any] = {key: value for key, value in masks.items() if key != "mask"}
        for modality, value in legacy_nested.items():
            key = modality if str(modality).endswith("_mask") else f"{modality}_mask"
            flattened[key] = value
        return flattened

    def _normalized_masks(self) -> dict[str, Any] | None:
        """Return flat masks from `Batch.masks` or fallback to `meta['masks']`."""
        if isinstance(self.masks, dict):
            return self._flatten_legacy_masks(self.masks)

        if not self.meta:
            return None

        legacy = self.meta.get("masks")
        if isinstance(legacy, dict):
            return self._flatten_legacy_masks(legacy)

        return None

    def get_masks(self, mask_name: str | None = None) -> Any | dict[str, Any] | None:
        """Return all masks or one specific mask by name."""
        masks = self._normalized_masks()
        if masks is None:
            return None

        if mask_name is None:
            return masks

        return masks.get(mask_name)
