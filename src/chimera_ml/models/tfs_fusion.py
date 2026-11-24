from __future__ import annotations

import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models._utils import _attention, _features, _masked_softmax, _mean, _projections
from chimera_ml.models.base import BaseModel

Pair = tuple[str, str]


def _pair_name(pair: Pair) -> str:
    return f"{pair[0]}->{pair[1]}"


def _normalize_groups(
    names: tuple[str, ...], groups: tuple[tuple[Pair, ...], ...] | None
) -> tuple[tuple[Pair, ...], ...]:
    if groups is None:
        pairs = tuple((left, right) for left in names for right in names if left != right)
        return (pairs,)
    normalized = tuple(tuple(tuple(pair) for pair in group) for group in groups)
    if not normalized or any(not group for group in normalized):
        raise ValueError("groups must contain at least one non-empty group.")

    known = set(names)
    seen: set[Pair] = set()
    for group in normalized:
        for pair in group:
            if len(pair) != 2 or pair[0] == pair[1] or pair[0] not in known or pair[1] not in known:
                raise ValueError(f"Invalid directed pair {pair!r} in groups.")
            if pair in seen:
                raise ValueError(f"Directed pair {_pair_name(pair)!r} occurs more than once in groups.")
            seen.add(pair)
    return normalized


def _directed_pair(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    query, query_valid = left
    key, key_valid = right
    attended = _attention(query, key, key_valid)
    value = _mean(query + attended, query_valid)
    pair_valid = query_valid.any(1) & key_valid.any(1)
    return value * pair_valid.to(value.dtype).unsqueeze(-1), pair_valid


def _collect_pairs(
    feats: dict[str, tuple[torch.Tensor, torch.Tensor]], groups: tuple[tuple[Pair, ...], ...], pair_fn
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], tuple[str, ...]]:
    values: dict[str, torch.Tensor] = {}
    valid: dict[str, torch.Tensor] = {}
    pair_keys = tuple(_pair_name(pair) for group in groups for pair in group)
    for key in pair_keys:
        left, right = key.split("->", 1)
        if left not in feats or right not in feats:
            continue
        values[key], valid[key] = pair_fn(feats[left], feats[right])
    return values, valid, pair_keys


def _hierarchical_fusion(
    groups: tuple[tuple[Pair, ...], ...],
    values: dict[str, torch.Tensor],
    valid: dict[str, torch.Tensor],
    group_layers: nn.ModuleList,
    output: nn.Linear,
    gate: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not values:
        raise ValueError("Fusion requires at least one configured directed pair.")

    first_value = next(iter(values.values()))
    pair_keys = tuple(_pair_name(pair) for group in groups for pair in group if _pair_name(pair) in values)
    pair_weights = first_value.new_zeros(first_value.shape[0], len(pair_keys))
    pair_available = torch.zeros_like(pair_weights, dtype=torch.bool)
    group_values: list[torch.Tensor] = []
    group_validities: list[torch.Tensor] = []
    pair_indices = {key: index for index, key in enumerate(pair_keys)}

    for layer, group in zip(group_layers, groups, strict=True):
        keys = [_pair_name(pair) for pair in group if _pair_name(pair) in values]
        if not keys:
            continue
        stacked = torch.stack([values[key] for key in keys], dim=1)
        available = torch.stack([valid[key] for key in keys], dim=1)
        if gate is None:
            weights = available.to(stacked.dtype)
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1)
        else:
            scores = gate(stacked).squeeze(-1)
            weights = _masked_softmax(scores, available)
        for index, key in enumerate(keys):
            pair_available[:, pair_indices[key]] = available[:, index]
            pair_weights[:, pair_indices[key]] = weights[:, index]

        group_valid = available.any(1)
        group_value = layer((stacked * weights.unsqueeze(-1)).sum(1))
        group_values.append(group_value * group_valid.to(group_value.dtype).unsqueeze(-1))
        group_validities.append(group_valid)

    pair_weights = pair_weights * pair_available.to(pair_weights.dtype)
    pair_weights = pair_weights / pair_weights.sum(1, keepdim=True).clamp_min(1)

    if not group_values:
        raise ValueError("Fusion has no directed pair available in batch.inputs.")

    stacked_groups = torch.stack(group_values, dim=1)
    group_available = torch.stack(group_validities, dim=1)
    group_weights = group_available.to(stacked_groups.dtype)
    group_weights = group_weights / group_weights.sum(1, keepdim=True).clamp_min(1)
    pre_output = (stacked_groups * group_weights.unsqueeze(-1)).sum(1)
    has_group = group_available.any(1)
    fused = output(pre_output) * has_group.to(pre_output.dtype).unsqueeze(-1)
    return fused, pair_weights, group_weights


class TFSFusionModel(BaseModel):
    """Hierarchical fusion of directed pairwise cross-modal interactions.

    The exact publication grouping is not available in this repository. The
    implementation therefore exposes explicit directed-pair groups as a
    testable generalization: pairs are averaged within each group, each group
    is transformed independently, and valid groups are averaged at the final
    hierarchy level.
    """

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        shared_dim: int,
        head: nn.Module,
        input_dims: dict[str, int] | None = None,
        groups: tuple[tuple[tuple[str, str], ...], ...] | None = None,
    ):
        super().__init__()
        self.names = tuple(encoders)
        self.encoders = nn.ModuleDict(encoders)
        self.shared_dim = shared_dim
        input_dims = input_dims or {}
        self.proj = _projections(self.names, input_dims, shared_dim)
        self.head = head
        self.groups = _normalize_groups(self.names, groups)
        self.group_layers = nn.ModuleList(nn.Linear(shared_dim, shared_dim) for _ in self.groups)
        self.output = nn.Linear(shared_dim, shared_dim)

    def _pair(
        self,
        left: tuple[torch.Tensor, torch.Tensor],
        right: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _directed_pair(left, right)

    def forward(self, batch: Batch) -> ModelOutput:
        feats = _features(self, batch, self.names)
        values, valid, _ = _collect_pairs(feats, self.groups, self._pair)
        if not values:
            raise ValueError("TFSFusionModel needs at least two available modalities.")

        fused, pair_weights, group_weights = _hierarchical_fusion(
            self.groups, values, valid, self.group_layers, self.output
        )
        aux: dict[str, torch.Tensor] = {}
        for key, value in values.items():
            aux[f"pair_{key}"] = value
            aux[f"pair_valid_{key}"] = valid[key]
        aux["pair_weights"] = pair_weights
        aux["group_weights"] = group_weights
        aux["fused"] = fused
        return ModelOutput(preds=self.head(fused), aux=aux)


class TFSDCMMOEFusionModel(BaseModel):
    """Adaptive hierarchical TFS with input-dependent directed-pair experts."""

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        shared_dim: int,
        head: nn.Module,
        input_dims: dict[str, int] | None = None,
        groups: tuple[tuple[tuple[str, str], ...], ...] | None = None,
        gate_hidden: int = 64,
    ):
        super().__init__()
        self.names = tuple(encoders)
        self.encoders = nn.ModuleDict(encoders)
        self.shared_dim = shared_dim
        input_dims = input_dims or {}
        self.proj = _projections(self.names, input_dims, shared_dim)
        self.head = head
        self.groups = _normalize_groups(self.names, groups)
        self.group_layers = nn.ModuleList(nn.Linear(shared_dim, shared_dim) for _ in self.groups)
        self.gates = nn.Sequential(nn.Linear(shared_dim, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
        self.output = nn.Linear(shared_dim, shared_dim)

    def _pair(
        self,
        left: tuple[torch.Tensor, torch.Tensor],
        right: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _directed_pair(left, right)

    def forward(self, batch: Batch) -> ModelOutput:
        feats = _features(self, batch, self.names)
        values, valid, _ = _collect_pairs(feats, self.groups, self._pair)
        if not values:
            raise ValueError("TFSDCMMOEFusionModel needs at least two available modalities.")

        fused, pair_weights, group_weights = _hierarchical_fusion(
            self.groups, values, valid, self.group_layers, self.output, gate=self.gates
        )
        aux: dict[str, torch.Tensor] = {}
        for key, value in values.items():
            aux[f"pair_{key}"] = value
            aux[f"pair_valid_{key}"] = valid[key]
        aux["pair_weights"] = pair_weights
        aux["group_weights"] = group_weights
        aux["fused"] = fused
        return ModelOutput(preds=self.head(fused), aux=aux)


@MODELS.register("tfs_fusion_model")
def tfs_fusion_model(**kwargs) -> TFSFusionModel:
    return TFSFusionModel(**kwargs)


@MODELS.register("tfs_dcmmmoe_fusion_model")
def tfs_dcmmmoe_fusion_model(**kwargs) -> TFSDCMMOEFusionModel:
    return TFSDCMMOEFusionModel(**kwargs)
