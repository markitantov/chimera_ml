from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import COLLATES

_TARGET_KEYS: tuple[str, ...] = ("targets", "target")


@dataclass
class MultimodalSequenceCollate:
    """Collate feature vectors and independent multimodal feature sequences.

    Rank-one inputs are sample-level feature vectors and retain shape ``[B, D]``.
    Rank-two inputs are temporal feature sequences and are padded independently
    per modality to ``[B, T_max, D]``. Dedicated ``<modality>_temporal_mask``
    masks identify valid temporal positions; ``<modality>_mask`` keeps the
    sample-presence meaning used by the legacy collator.

    Missing modality rows are zero-filled. ``pad_value`` is used only for
    positions beyond the length of a present temporal sequence.
    """

    pad_sequences: bool = True
    include_legacy_meta_masks: bool = False
    pad_value: float = 0.0

    def __call__(self, batch: list[dict[str, Any]]) -> Batch:
        """Build a batch with independent modality padding and temporal masks."""
        if not batch:
            raise ValueError("MultimodalSequenceCollate received an empty batch.")

        inputs, presence_masks, temporal_masks = self._collate_inputs(batch)
        targets = self._collate_targets(batch)

        masks: dict[str, torch.Tensor] = {}
        for modality in inputs:
            masks[f"{modality}_mask"] = presence_masks[modality]
            masks[f"{modality}_temporal_mask"] = temporal_masks[modality]

        sample_meta = [dict(sample.get("meta", {})) for sample in batch]
        meta: dict[str, Any] = {"sample_meta": sample_meta}
        if self.include_legacy_meta_masks:
            meta["masks"] = masks

        return Batch(inputs=inputs, targets=targets, masks=masks, meta=meta)

    def _collate_inputs(
        self,
        batch: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        modality_names = sorted({modality for sample in batch for modality in dict(sample.get("inputs", {}))})
        inputs: dict[str, torch.Tensor] = {}
        presence_masks: dict[str, torch.Tensor] = {}
        temporal_masks: dict[str, torch.Tensor] = {}

        for modality in modality_names:
            tensors = [self._as_tensor_or_none(dict(sample.get("inputs", {})).get(modality)) for sample in batch]
            present = [tensor for tensor in tensors if tensor is not None]
            if not present:
                continue

            rank = present[0].ndim
            if rank not in (1, 2):
                raise ValueError(
                    f"Modality '{modality}' must contain feature vectors [D] or sequences [T, D], got rank {rank}."
                )
            if any(tensor.ndim != rank for tensor in present):
                raise ValueError(f"Modality '{modality}' has incompatible ranks across samples.")

            feature_dim = present[0].shape[-1] if rank == 2 else present[0].shape[0]
            if any((tensor.shape[-1] if rank == 2 else tensor.shape[0]) != feature_dim for tensor in present):
                raise ValueError(f"Modality '{modality}' has incompatible feature dimensions across samples.")

            explicit_masks = [self._get_explicit_temporal_mask(sample, modality) for sample in batch]
            if rank == 1:
                collated, temporal = self._collate_vectors(tensors, feature_dim, explicit_masks, modality)
            else:
                collated, temporal = self._collate_sequences(tensors, feature_dim, explicit_masks, modality)

            inputs[modality] = collated
            presence_masks[modality] = torch.tensor([tensor is not None for tensor in tensors], dtype=torch.float32)
            temporal_masks[modality] = temporal

        return inputs, presence_masks, temporal_masks

    def _collate_vectors(
        self,
        tensors: Sequence[torch.Tensor | None],
        feature_dim: int,
        explicit_masks: Sequence[torch.Tensor | None],
        modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = next(tensor for tensor in tensors if tensor is not None)
        output = reference.new_zeros((len(tensors), feature_dim))
        temporal = torch.zeros((len(tensors), 1), dtype=torch.bool)
        for index, tensor in enumerate(tensors):
            if tensor is not None:
                output[index] = tensor
                temporal[index, 0] = True

            explicit = explicit_masks[index]
            if explicit is not None:
                self._validate_explicit_mask(explicit, 1, modality)
                temporal[index, 0] &= bool(explicit[0]) if tensor is not None else False

        return output, temporal

    def _collate_sequences(
        self,
        tensors: Sequence[torch.Tensor | None],
        feature_dim: int,
        explicit_masks: Sequence[torch.Tensor | None],
        modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        present = [tensor for tensor in tensors if tensor is not None]
        lengths = [int(tensor.shape[0]) for tensor in present]
        if not self.pad_sequences and len(set(lengths)) > 1:
            raise ValueError(
                f"Modality '{modality}' has variable temporal lengths with pad_sequences=False. "
                "Enable padding for variable-length data."
            )

        max_length = max(lengths, default=0)
        reference = present[0]
        output = reference.new_full((len(tensors), max_length, feature_dim), fill_value=self.pad_value)
        temporal = torch.zeros((len(tensors), max_length), dtype=torch.bool)
        for index, tensor in enumerate(tensors):
            explicit = explicit_masks[index]
            length = 0 if tensor is None else int(tensor.shape[0])
            if explicit is not None:
                self._validate_explicit_mask(explicit, length, modality)

            if tensor is None:
                continue

            if length:
                output[index, :length] = tensor
                temporal[index, :length] = True
            if explicit is not None and length:
                temporal[index, :length] &= explicit

        missing = [index for index, tensor in enumerate(tensors) if tensor is None]
        if missing:
            output[missing] = 0
        return output, temporal

    def _collate_targets(self, batch: Sequence[Mapping[str, Any]]) -> torch.Tensor | None:
        tensors = [self._get_target_tensor(sample) for sample in batch]
        if all(tensor is None for tensor in tensors):
            return None

        present = [tensor for tensor in tensors if tensor is not None]
        reference = present[0]
        if any(tensor.ndim != reference.ndim for tensor in present):
            raise ValueError("Cannot collate targets with different ranks.")
        if any(tuple(tensor.shape[1:]) != tuple(reference.shape[1:]) for tensor in present):
            raise ValueError("Cannot collate targets with incompatible feature dimensions.")

        if self.pad_sequences and reference.ndim >= 1:
            max_length = max(int(tensor.shape[0]) for tensor in present)
            output = reference.new_full((len(tensors), max_length, *reference.shape[1:]), fill_value=self.pad_value)
            for index, tensor in enumerate(tensors):
                if tensor is not None:
                    output[index, : tensor.shape[0]] = tensor
            return output

        output = reference.new_full((len(tensors), *reference.shape), fill_value=self.pad_value)
        for index, tensor in enumerate(tensors):
            if tensor is None:
                continue
            if tuple(tensor.shape) != tuple(reference.shape):
                raise ValueError(
                    "Cannot stack targets with different shapes when pad_sequences=False. "
                    "Enable padding for variable-length data."
                )
            output[index] = tensor
        return output

    @staticmethod
    def _validate_explicit_mask(mask: torch.Tensor, expected_length: int, modality: str) -> None:
        if mask.ndim != 1 or mask.shape[0] != expected_length:
            raise ValueError(
                f"Explicit mask for '{modality}' must have shape ({expected_length},), got {tuple(mask.shape)}."
            )

    @staticmethod
    def _get_explicit_temporal_mask(sample: Mapping[str, Any], modality: str) -> torch.Tensor | None:
        key = f"{modality}_temporal_mask"
        meta = sample.get("meta")
        candidates: list[Any] = []
        if isinstance(meta, Mapping):
            candidates.append(meta.get(key))
            nested = meta.get("masks")
            if isinstance(nested, Mapping):
                candidates.append(nested.get(key))
        nested = sample.get("masks")
        if isinstance(nested, Mapping):
            candidates.append(nested.get(key))

        value = next((candidate for candidate in candidates if candidate is not None), None)
        if value is None:
            return None
        try:
            return torch.as_tensor(value).bool()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Explicit mask for '{modality}' must be a one-dimensional boolean-like tensor.") from exc

    @staticmethod
    def _as_tensor_or_none(value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        return value if torch.is_tensor(value) else torch.as_tensor(value)

    @staticmethod
    def _get_target_tensor(sample: Mapping[str, Any]) -> torch.Tensor | None:
        for key in _TARGET_KEYS:
            if key in sample:
                return MultimodalSequenceCollate._as_tensor_or_none(sample.get(key))
        return None


@COLLATES.register("multimodal_sequence_collate")
def multimodal_sequence_collate(**params: Any) -> MultimodalSequenceCollate:
    """Create ``MultimodalSequenceCollate`` from registry configuration."""
    return MultimodalSequenceCollate(**params)
