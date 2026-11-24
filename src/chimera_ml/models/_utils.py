from __future__ import annotations

import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch


def _mask(batch: Batch, name: str, x: torch.Tensor) -> torch.Tensor:
    """Return valid positions for a normalized ``[B, T, D]`` modality sequence.

    A modality mask may describe either sample presence (``[B]``) or temporal
    validity (``[B, T]``). ``MaskingCollate`` also emits a shared sequence mask;
    it is combined only when its temporal width matches this modality. A width
    mismatch is left untouched because the shared mask does not identify which
    modality-specific positions are padded.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected normalized features for '{name}' to have shape [B, T, D], got {tuple(x.shape)}.")

    batch_size, time_steps = x.shape[:2]
    value = batch.get_masks(f"{name}_mask")
    if value is None:
        result = torch.ones((batch_size, time_steps), dtype=torch.bool, device=x.device)
    else:
        result = torch.as_tensor(value, device=x.device).bool()
        if result.ndim == 1:
            if result.shape[0] != batch_size:
                raise ValueError(f"Mask for '{name}' has batch dimension {result.shape[0]}, expected {batch_size}.")
            result = result[:, None].expand(-1, time_steps)
        elif result.ndim == 2:
            if result.shape[0] != batch_size:
                raise ValueError(f"Mask for '{name}' has batch dimension {result.shape[0]}, expected {batch_size}.")
            if result.shape[1] != time_steps:
                raise ValueError(f"Mask for '{name}' must have temporal width {time_steps}, got {result.shape[1]}.")
        else:
            raise ValueError(f"Mask for '{name}' must have shape (B,) or (B, T), got {tuple(result.shape)}.")

    sequence_mask = batch.get_masks("sequence_mask")
    if sequence_mask is not None:
        sequence_mask = torch.as_tensor(sequence_mask, device=x.device).bool()
        if sequence_mask.ndim == 1:
            if sequence_mask.shape[0] != batch_size:
                raise ValueError(f"sequence_mask has batch dimension {sequence_mask.shape[0]}, expected {batch_size}.")
            sequence_mask = sequence_mask[:, None]
        elif sequence_mask.ndim == 2:
            if sequence_mask.shape[0] != batch_size:
                raise ValueError(f"sequence_mask has batch dimension {sequence_mask.shape[0]}, expected {batch_size}.")
        else:
            raise ValueError(f"sequence_mask must have shape (B,) or (B, T), got {tuple(sequence_mask.shape)}.")

        if sequence_mask.shape == result.shape:
            result = result & sequence_mask

    return result


def _sequence(x: torch.Tensor) -> torch.Tensor:
    """Normalize a feature tensor from ``[B, D]`` or ``[B, T, D]`` to 3-D."""
    if not torch.is_tensor(x):
        raise ValueError(f"Expected [B, D] or [B, T, D], got {type(x).__name__}.")
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected [B, D] or [B, T, D], got {tuple(x.shape)}.")


def _mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Pool a sequence over valid positions, returning zero for empty rows."""
    if x.ndim != 3:
        raise ValueError(f"Expected values with shape [B, T, D], got {tuple(x.shape)}.")
    valid = torch.as_tensor(valid, device=x.device).bool()
    if valid.shape != x.shape[:2]:
        raise ValueError(f"Expected validity mask with shape {tuple(x.shape[:2])}, got {tuple(valid.shape)}.")

    weight = valid.to(x.dtype).unsqueeze(-1)
    denominator = weight.sum(1)
    return (x * weight).sum(1) / denominator.clamp_min(1)


def _masked_softmax(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compute a masked softmax with an all-invalid result of exactly zero."""
    if scores.ndim < 2:
        raise ValueError(f"Expected scores with at least two dimensions, got {tuple(scores.shape)}.")
    valid = torch.as_tensor(valid, device=scores.device).bool()
    if valid.ndim != 2 or valid.shape[0] != scores.shape[0] or valid.shape[1] != scores.shape[-1]:
        raise ValueError(
            f"Expected validity mask with shape ({scores.shape[0]}, {scores.shape[-1]}), got {tuple(valid.shape)}."
        )

    work_dtype = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
    work_scores = scores.to(work_dtype)
    expanded_valid = valid
    while expanded_valid.ndim < work_scores.ndim:
        expanded_valid = expanded_valid.unsqueeze(1)

    masked_scores = work_scores.masked_fill(~expanded_valid, -torch.inf)
    maximum = masked_scores.amax(dim=-1, keepdim=True)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exponent = torch.exp(masked_scores - safe_maximum)
    denominator = exponent.sum(dim=-1, keepdim=True)
    weights = exponent / denominator.clamp_min(1)
    return weights.to(scores.dtype)


def _attention(query: torch.Tensor, key: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
    """Apply scaled dot-product attention with safe handling of empty key rows."""
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError(
            f"Expected query and key with shape [B, T, D], got {tuple(query.shape)} and {tuple(key.shape)}."
        )
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError(
            "Query and key must have matching batch and feature dimensions, "
            f"got {tuple(query.shape)} and {tuple(key.shape)}."
        )

    key_valid = torch.as_tensor(key_valid, device=query.device).bool()
    if key_valid.shape != key.shape[:2]:
        raise ValueError(f"Expected key validity mask with shape {tuple(key.shape[:2])}, got {tuple(key_valid.shape)}.")

    work_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    query_work = query.to(work_dtype)
    key_work = key.to(work_dtype)
    scores = query_work @ key_work.transpose(-1, -2) * query.shape[-1] ** -0.5
    weights = _masked_softmax(scores, key_valid)
    return (weights.to(work_dtype) @ key_work).to(query.dtype)


def _features(model: nn.Module, batch: Batch, names: tuple[str, ...]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Encode, project and validate the requested modality feature sequences."""
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    batch_size: int | None = None
    for name in names:
        if name not in batch.inputs:
            continue

        encoded = model.encoders[name](batch.inputs[name])
        x = _sequence(encoded)
        if batch_size is None:
            batch_size = x.shape[0]
        elif x.shape[0] != batch_size:
            raise ValueError(f"Modality '{name}' has batch dimension {x.shape[0]}, expected {batch_size}.")

        projected = model.proj[name](x)
        if not torch.is_tensor(projected) or projected.ndim != 3:
            shape = getattr(projected, "shape", None)
            raise ValueError(f"Projection for '{name}' must return [B, T, D], got {shape}.")
        if projected.shape[-1] != model.shared_dim:
            raise ValueError(f"Feature '{name}' has width {projected.shape[-1]}, expected {model.shared_dim}.")

        valid = _mask(batch, name, projected)
        result[name] = (projected, valid)

    if not result:
        raise ValueError("No configured modalities are present in batch.inputs.")

    return result


def _projections(names: tuple[str, ...], input_dims: dict[str, int], shared_dim: int) -> nn.ModuleDict:
    """Build one projection per modality, preserving identity for preprojected inputs."""
    if shared_dim <= 0:
        raise ValueError(f"shared_dim must be positive, got {shared_dim}.")

    projections = {}
    for name in names:
        if name in input_dims:
            if input_dims[name] <= 0:
                raise ValueError(f"Input dimension for '{name}' must be positive, got {input_dims[name]}.")
            projections[name] = nn.Linear(input_dims[name], shared_dim)
        else:
            projections[name] = nn.Identity()
    return nn.ModuleDict(projections)
