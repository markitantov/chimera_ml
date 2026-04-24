from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import COLLATES

_TARGET_KEYS: tuple[str, ...] = ("targets", "target")


@dataclass
class MaskingCollate:
    """Collate multimodal samples with variable sequence lengths and modality masks."""

    pad_sequences: bool = True
    include_legacy_meta_masks: bool = False
    pad_value: float = 0.0

    def __call__(self, batch: list[dict[str, Any]]) -> Batch:
        """Build a `Batch` with padded tensors, `sequence_mask`, and `{modality}_mask` flags."""
        if not batch:
            raise ValueError("MaskingCollate received an empty batch.")

        inputs, input_lengths, modality_masks = self._collate_inputs(batch)
        targets, target_lengths = self._collate_targets(batch)
        sequence_mask = self._build_sequence_mask(batch, input_lengths, target_lengths)

        masks: dict[str, torch.Tensor] = {"sequence_mask": sequence_mask}
        masks.update(modality_masks)

        sample_meta = [dict(sample.get("meta", {})) for sample in batch]
        meta: dict[str, Any] = {"sample_meta": sample_meta}
        if self.include_legacy_meta_masks:
            meta["masks"] = masks

        return Batch(inputs=inputs, targets=targets, masks=masks, meta=meta)

    def _collate_inputs(
        self,
        batch: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Pad/stack modality tensors and produce modality-presence masks."""
        modality_names = sorted({modality for sample in batch for modality in dict(sample.get("inputs", {}))})

        batch_size = len(batch)
        inputs: dict[str, torch.Tensor] = {}
        input_lengths: dict[str, torch.Tensor] = {}
        modality_masks: dict[str, torch.Tensor] = {}

        for modality in modality_names:
            tensors: list[torch.Tensor | None] = []
            lengths = torch.zeros(batch_size, dtype=torch.long)
            presence = torch.zeros(batch_size, dtype=torch.float32)

            for idx, sample in enumerate(batch):
                raw_inputs = dict(sample.get("inputs", {}))
                value = raw_inputs.get(modality)
                tensor = self._as_tensor_or_none(value)
                tensors.append(tensor)

                if tensor is None:
                    continue

                presence[idx] = 1.0
                lengths[idx] = self._tensor_sequence_length(tensor)

            collated = self._pad_or_stack(tensors)
            if collated is None:
                continue

            inputs[modality] = collated
            input_lengths[modality] = lengths
            modality_masks[f"{modality}_mask"] = presence

        return inputs, input_lengths, modality_masks

    def _collate_targets(
        self,
        batch: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Pad/stack targets and return per-sample target lengths."""
        tensors: list[torch.Tensor | None] = []
        lengths = torch.zeros(len(batch), dtype=torch.long)

        for idx, sample in enumerate(batch):
            target = self._get_target_tensor(sample)
            tensors.append(target)
            if target is not None:
                lengths[idx] = self._tensor_sequence_length(target)

        if all(tensor is None for tensor in tensors):
            return None, lengths

        return self._pad_or_stack(tensors), lengths

    def _build_sequence_mask(
        self,
        batch: Sequence[Mapping[str, Any]],
        input_lengths: Mapping[str, torch.Tensor],
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Create sequence mask from explicit meta masks or inferred sample lengths."""
        inferred_lengths = self._infer_sequence_lengths(input_lengths, target_lengths)
        explicit_masks = self._extract_explicit_sequence_masks(batch)

        max_len = int(inferred_lengths.max().item()) if inferred_lengths.numel() else 0
        for mask in explicit_masks:
            if mask is not None:
                max_len = max(max_len, int(mask.numel()))

        sequence_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

        for idx, explicit_mask in enumerate(explicit_masks):
            if explicit_mask is not None:
                n = min(int(explicit_mask.numel()), max_len)
                sequence_mask[idx, :n] = explicit_mask[:n]
                continue

            length = int(inferred_lengths[idx].item())
            if length > 0:
                sequence_mask[idx, :length] = True

        return sequence_mask

    def _infer_sequence_lengths(
        self,
        input_lengths: Mapping[str, torch.Tensor],
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Infer sequence lengths from inputs, fallback to targets when inputs are absent."""
        if input_lengths:
            stacked_inputs = torch.stack(list(input_lengths.values()), dim=0)
            inferred = torch.max(stacked_inputs, dim=0).values
        elif target_lengths.numel() > 0:
            inferred = target_lengths.clone()
        else:
            return torch.zeros(0, dtype=torch.long)

        # Keep mask width >= 1 for completely empty samples.
        return torch.clamp(inferred, min=1)

    @staticmethod
    def _extract_explicit_sequence_masks(
        batch: Sequence[Mapping[str, Any]],
    ) -> list[torch.Tensor | None]:
        """Extract optional per-sample explicit sequence masks from sample meta."""
        extracted: list[torch.Tensor | None] = []
        for sample in batch:
            meta = sample.get("meta")
            if not isinstance(meta, Mapping):
                extracted.append(None)
                continue

            masks = meta.get("masks")
            if not isinstance(masks, Mapping):
                extracted.append(None)
                continue

            raw_sequence_mask = masks.get("sequence_mask")
            if raw_sequence_mask is None:
                extracted.append(None)
                continue

            extracted.append(torch.as_tensor(raw_sequence_mask).to(dtype=torch.bool).flatten())

        return extracted

    def _pad_or_stack(self, tensors: Sequence[torch.Tensor | None]) -> torch.Tensor | None:
        """Pad the first dimension for variable-length tensors, otherwise stack."""
        present = [tensor for tensor in tensors if tensor is not None]
        if not present:
            return None

        reference = present[0]
        if any(tensor.ndim != reference.ndim for tensor in present):
            raise ValueError("Cannot collate tensors with different ranks.")

        if any(tuple(tensor.shape[1:]) != tuple(reference.shape[1:]) for tensor in present):
            raise ValueError("Cannot collate tensors with incompatible feature dimensions.")

        batch_size = len(tensors)
        if self.pad_sequences and reference.ndim >= 1:
            max_len = max(int(tensor.shape[0]) for tensor in present)
            output = reference.new_full(
                (batch_size, max_len, *reference.shape[1:]),
                fill_value=self.pad_value,
            )
            for idx, tensor in enumerate(tensors):
                if tensor is None:
                    continue

                length = int(tensor.shape[0])
                output[idx, :length] = tensor

            return output

        output = reference.new_full((batch_size, *reference.shape), fill_value=self.pad_value)
        for idx, tensor in enumerate(tensors):
            if tensor is None:
                continue

            if tuple(tensor.shape) != tuple(reference.shape):
                raise ValueError(
                    "Cannot stack tensors with different shapes when pad_sequences=False. "
                    "Enable pad_sequences for variable-length data."
                )

            output[idx] = tensor

        return output

    @staticmethod
    def _tensor_sequence_length(tensor: torch.Tensor) -> int:
        """Return sequence length proxy for a tensor sample."""
        return int(tensor.shape[0]) if tensor.ndim >= 1 else 1

    @staticmethod
    def _as_tensor_or_none(value: Any) -> torch.Tensor | None:
        """Convert input value to tensor unless it is explicitly missing."""
        if value is None:
            return None
        return value if torch.is_tensor(value) else torch.as_tensor(value)

    @staticmethod
    def _get_target_tensor(sample: Mapping[str, Any]) -> torch.Tensor | None:
        """Read target tensor from a sample supporting `target` and `targets` keys."""
        for key in _TARGET_KEYS:
            if key in sample:
                return MaskingCollate._as_tensor_or_none(sample.get(key))

        return None


@COLLATES.register("masking_collate")
def masking_collate(**params: Any) -> MaskingCollate:
    """Create `MaskingCollate` from registry configuration."""
    return MaskingCollate(**params)
