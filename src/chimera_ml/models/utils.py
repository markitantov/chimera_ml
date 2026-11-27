import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch


def get_modality_mask(batch: Batch, name: str, values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected normalized projected features for '{name}' to have shape [B, T, D].")

    batch_size, time_steps = values.shape[:2]
    dedicated = batch.get_masks(f"{name}_temporal_mask")
    if dedicated is not None:
        result = torch.as_tensor(dedicated, device=values.device).bool()
        if result.shape != (batch_size, time_steps):
            raise ValueError(f"Temporal mask for '{name}' must have shape {(batch_size, time_steps)}.")
        return result

    value = batch.get_masks(f"{name}_mask")
    if value is None:
        result = torch.ones((batch_size, time_steps), dtype=torch.bool, device=values.device)
    else:
        result = torch.as_tensor(value, device=values.device).bool()
        if result.ndim == 1 and result.shape[0] == batch_size:
            result = result[:, None].expand(-1, time_steps)
        elif result.shape != (batch_size, time_steps):
            raise ValueError(f"Mask for '{name}' must have shape (B,) or (B, T).")

    sequence = batch.get_masks("sequence_mask")
    if sequence is not None:
        sequence = torch.as_tensor(sequence, device=values.device).bool()
        if sequence.ndim == 1 and sequence.shape[0] == batch_size:
            sequence = sequence[:, None]
        if sequence.shape == result.shape:
            result = result & sequence

    return result


def normalize_sequence(values: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(values):
        raise ValueError(f"Expected [B, D] or [B, T, D], got {type(values).__name__}.")

    if values.ndim == 2:
        return values[:, None, :]

    if values.ndim == 3:
        return values

    raise ValueError(f"Expected [B, D] or [B, T, D], got {tuple(values.shape)}.")


def get_modality_validity(batch: Batch, name: str) -> torch.Tensor:
    values = batch.inputs.get(name)
    if values is None:
        return torch.zeros(0, dtype=torch.bool)

    return get_modality_mask(batch, name, normalize_sequence(values)).any(dim=1)


def prepare_projected_features(
    model: nn.Module, batch: Batch, names: tuple[str, ...]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    batch_size: int | None = None

    for name in names:
        if name not in batch.inputs:
            continue

        values = normalize_sequence(batch.inputs[name])
        if batch_size is not None and values.shape[0] != batch_size:
            raise ValueError(f"Modality '{name}' has an incompatible batch dimension.")

        batch_size = values.shape[0]
        expected = model.input_dims[name]
        if values.shape[-1] != expected:
            raise ValueError(f"Feature '{name}' has width {values.shape[-1]}, expected {expected}.")

        projected = model.proj[name](values)
        result[name] = projected, get_modality_mask(batch, name, projected)

    if not result:
        raise ValueError("No configured modalities are present in batch.inputs.")

    return result


def validate_input_dimensions(input_dims: dict[str, int]) -> dict[str, int]:
    if not isinstance(input_dims, dict) or not input_dims:
        raise ValueError("input_dims must be a non-empty dictionary.")

    if any(not isinstance(name, str) or not name for name in input_dims):
        raise ValueError("input_dims keys must be non-empty strings.")

    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in input_dims.values()):
        raise ValueError("input_dims values must be positive integers.")

    return dict(input_dims)


def build_projections(
    names: tuple[str, ...],
    input_dims: dict[str, int],
    shared_dim: int,
) -> nn.ModuleDict:
    if shared_dim <= 0:
        raise ValueError(f"shared_dim must be positive, got {shared_dim}.")

    validate_input_dimensions(input_dims)
    missing = [name for name in names if name not in input_dims]
    if missing:
        raise ValueError(f"Missing input dimensions for modalities: {missing!r}.")

    return nn.ModuleDict({name: nn.Linear(input_dims[name], shared_dim) for name in names})
