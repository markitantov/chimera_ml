import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch


def _mask(batch: Batch, name: str, x: torch.Tensor) -> torch.Tensor:
    """Return the valid positions for a modality's normalized sequence."""
    value = batch.get_masks(f"{name}_mask")
    if value is None:
        result = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
    else:
        result = torch.as_tensor(value, device=x.device).bool()
        if result.ndim == 1:
            result = result[:, None]
        if result.ndim != 2:
            raise ValueError(f"Mask for '{name}' must have shape (B,) or (B, T), got {tuple(result.shape)}.")
        if result.shape[1] == 1 and x.shape[1] != 1:
            result = result.expand(-1, x.shape[1])
        if result.shape != x.shape[:-1]:
            raise ValueError(f"Mask for '{name}' must have shape (B,) or (B, T), got {tuple(result.shape)}.")

    sequence_mask = batch.get_masks("sequence_mask")
    if sequence_mask is not None:
        sequence_mask = torch.as_tensor(sequence_mask, device=x.device).bool()
        if sequence_mask.ndim == 1:
            sequence_mask = sequence_mask[:, None]
        # MaskingCollate emits a shared sequence mask. Apply it when this
        # modality has the same temporal width; a 2-D modality normalized to
        # T=1 must not be broadcast over another modality's longer sequence.
        if sequence_mask.shape == result.shape:
            result = result & sequence_mask

    return result


def _sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected [B, D] or [B, T, D], got {tuple(x.shape)}.")


def _mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.to(x.dtype).unsqueeze(-1)
    return (x * weight).sum(1) / weight.sum(1).clamp_min(1)


def _attention(query: torch.Tensor, key: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
    scores = query @ key.transpose(-1, -2) * query.shape[-1] ** -0.5
    scores = scores.masked_fill(~key_valid[:, None, :], -torch.finfo(scores.dtype).max)
    weights = torch.softmax(scores, dim=-1) * key_valid[:, None, :].to(scores.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    return weights @ key


def _features(model: nn.Module, batch: Batch, names: tuple[str, ...]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    result = {}
    for name in names:
        if name not in batch.inputs:
            continue
        x = _sequence(model.proj[name](model.encoders[name](batch.inputs[name])))
        if x.shape[-1] != model.shared_dim:
            raise ValueError(f"Feature '{name}' has width {x.shape[-1]}, expected {model.shared_dim}.")
        result[name] = (x, _mask(batch, name, x))

    if not result:
        raise ValueError("No configured modalities are present in batch.inputs.")
    return result


def _projections(names: tuple[str, ...], input_dims: dict[str, int], shared_dim: int) -> nn.ModuleDict:
    return nn.ModuleDict(
        {name: nn.Linear(input_dims[name], shared_dim) if name in input_dims else nn.Identity() for name in names}
    )
