import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch


def get_modality_mask(batch: Batch, name: str, values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected normalized prepare_projected_features for '{name}' to have shape [B, T, D].")

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


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


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


def build_projections(names: tuple[str, ...], input_dims: dict[str, int], shared_dim: int) -> nn.ModuleDict:
    if shared_dim <= 0:
        raise ValueError(f"shared_dim must be positive, got {shared_dim}.")
    validate_input_dimensions(input_dims)
    missing = [name for name in names if name not in input_dims]
    if missing:
        raise ValueError(f"Missing input dimensions for modalities: {missing!r}.")
    return nn.ModuleDict({name: nn.Linear(input_dims[name], shared_dim) for name in names})


def masked_softmax(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if scores.ndim < 2:
        raise ValueError(f"Expected scores with at least two dimensions, got {tuple(scores.shape)}.")
    valid = torch.as_tensor(valid, device=scores.device).bool()
    if valid.ndim != 2 or valid.shape[0] != scores.shape[0] or valid.shape[1] != scores.shape[-1]:
        raise ValueError(
            f"Expected validity mask with shape ({scores.shape[0]}, {scores.shape[-1]}), got {tuple(valid.shape)}."
        )
    if scores.shape[-1] == 0:
        return scores * 0
    work_dtype = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
    work_scores = scores.to(work_dtype)
    expanded = valid
    while expanded.ndim < work_scores.ndim:
        expanded = expanded.unsqueeze(1)
    masked = work_scores.masked_fill(~expanded, -torch.inf)
    maximum = masked.amax(dim=-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    weights = torch.exp(masked - maximum)
    return (weights / weights.sum(dim=-1, keepdim=True).clamp_min(1)).to(scores.dtype)


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
    values, _ = scaled_dot_product_attention_with_weights(query, key, key_valid)
    return values


def scaled_dot_product_attention_with_weights(
    query: torch.Tensor, key: torch.Tensor, key_valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    work_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    query_work = query.to(work_dtype)
    key_work = key.to(work_dtype)
    scores = query_work @ key_work.transpose(-1, -2) * query.shape[-1] ** -0.5
    weights = masked_softmax(scores, key_valid)
    return (weights.to(work_dtype) @ key_work).to(query.dtype), weights
