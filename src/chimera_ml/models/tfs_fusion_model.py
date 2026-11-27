from collections.abc import Sequence

import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel
from chimera_ml.models.common_layers import (
    SafeBatchNorm1d,
    StatPool,
    TransformerCrossAttention,
)
from chimera_ml.models.utils import (
    build_projections,
    get_modality_validity,
    prepare_projected_features,
    validate_input_dimensions,
)


def pair_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}->{pair[1]}"


def normalize_groups(
    names: tuple[str, ...],
    groups: Sequence[Sequence[Sequence[str]]] | None,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    if groups is None:
        forward = tuple((left, right) for index, left in enumerate(names) for right in names[index + 1 :])
        groups = (
            forward,
            tuple((right, left) for left, right in forward),
        )

    normalized = tuple(tuple(tuple(pair) for pair in group) for group in groups)
    if not normalized or any(not group for group in normalized):
        raise ValueError("groups must contain at least one non-empty group.")

    known = set(names)
    seen: set[tuple[str, str]] = set()
    for group in normalized:
        for pair in group:
            if len(pair) != 2 or pair[0] == pair[1] or pair[0] not in known or pair[1] not in known:
                raise ValueError(f"Invalid directed pair {pair!r} in groups.")

            if pair in seen:
                raise ValueError(f"Directed pair {pair_name(pair)!r} occurs more than once in groups.")

            seen.add(pair)

    return normalized


class TFSFusionModel(BaseModel):
    """Hierarchical bidirectional TFS with Transformer cross-attention blocks."""

    def __init__(
        self,
        input_dims: dict[str, int],
        shared_dim: int,
        output_dim: int,
        groups: Sequence[Sequence[Sequence[str]]] | None = None,
        dropout: float = 0.1,
        max_position_length: int = 5000,
    ):
        super().__init__()
        self.input_dims = validate_input_dimensions(input_dims)
        if shared_dim <= 0:
            raise ValueError(f"shared_dim must be positive, got {shared_dim}.")

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")

        if max_position_length <= 0:
            raise ValueError(f"max_position_length must be positive, got {max_position_length}.")

        self.names = tuple(self.input_dims)
        self.shared_dim = shared_dim
        self.proj = build_projections(self.names, self.input_dims, shared_dim)
        self.groups = normalize_groups(self.names, groups)
        pairs = tuple(pair for group in self.groups for pair in group)
        self.pair_blocks = nn.ModuleDict(
            {pair_name(pair): TransformerCrossAttention(shared_dim, dropout, max_position_length) for pair in pairs}
        )
        self.group_blocks = nn.ModuleList(
            TransformerCrossAttention(shared_dim, dropout, max_position_length) for _ in self.groups
        )
        self.stat_pool = StatPool()
        self.group_projections = nn.ModuleList(nn.Linear(shared_dim * 2, shared_dim) for _ in self.groups)
        self.fusion = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            SafeBatchNorm1d(shared_dim),
            nn.Tanh(),
        )
        self.prediction = nn.Linear(shared_dim, output_dim)

    def _pair(
        self,
        query: tuple[torch.Tensor, torch.Tensor],
        context: tuple[torch.Tensor, torch.Tensor],
        key: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_values, query_valid = query
        context_values, context_valid = context
        result, weights = self.pair_blocks[key](
            query_values,
            context_values,
            query_valid,
            context_valid,
        )
        valid = query_valid & context_valid.any(dim=1, keepdim=True)
        result = result * valid.unsqueeze(-1).to(result.dtype)
        return result, valid, weights

    def _interactions(
        self,
        feature_map: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        values: dict[str, torch.Tensor] = {}
        valid: dict[str, torch.Tensor] = {}
        attention: dict[str, torch.Tensor] = {}
        for group in self.groups:
            for pair in group:
                key = pair_name(pair)
                if pair[0] not in feature_map or pair[1] not in feature_map:
                    continue

                values[key], valid[key], attention[key] = self._pair(
                    feature_map[pair[0]],
                    feature_map[pair[1]],
                    key,
                )

        return values, valid, attention

    def _fuse_groups(
        self,
        values: dict[str, torch.Tensor],
        valid: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not values:
            raise ValueError("TFSFusionModel needs at least two available modalities.")

        group_values: list[torch.Tensor] = []
        group_valid: list[torch.Tensor] = []
        group_attention: dict[str, torch.Tensor] = {}

        for group_index, group in enumerate(self.groups):
            keys = [pair_name(pair) for pair in group if pair_name(pair) in values]
            if not keys:
                continue

            query = torch.cat([values[key] for key in keys], dim=1)
            query_valid = torch.cat([valid[key] for key in keys], dim=1)
            other_keys = [
                pair_name(pair)
                for other_index, other in enumerate(self.groups)
                if other_index != group_index
                for pair in other
                if pair_name(pair) in values
            ]

            if other_keys:
                context = torch.cat([values[key] for key in other_keys], dim=1)
                context_valid = torch.cat([valid[key] for key in other_keys], dim=1)
                refined, coefficients = self.group_blocks[group_index](
                    query,
                    context,
                    query_valid,
                    context_valid,
                )
            else:
                refined = query * query_valid.unsqueeze(-1).to(query.dtype)
                coefficients = query.new_zeros(query.shape[0], query.shape[1], 0)

            pooled = self.stat_pool(refined, query_valid)
            current = self.group_projections[group_index](pooled)
            available = query_valid.any(dim=1)
            current = current * available.unsqueeze(-1).to(current.dtype)
            group_values.append(current)
            group_valid.append(available)
            group_attention[f"group_attention_{group_index}"] = coefficients

        if not group_values:
            raise ValueError("TFSFusionModel has no available directed pair.")

        stacked = torch.stack(group_values, dim=1)
        available = torch.stack(group_valid, dim=1)
        fused = self.stat_pool(stacked, available)
        count = available.sum(dim=1, keepdim=True).clamp_min(1)
        group_weights = available.to(stacked.dtype) / count
        return fused, group_weights, group_attention

    def validate_intervention(
        self,
        original: Batch,
        intervened: Batch,
        modality: str,
        policy: str,
    ) -> str | None:
        batch_size = next(iter(original.inputs.values())).shape[0]
        device = next(iter(original.inputs.values())).device
        supported = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for group in self.groups:
            for left, right in group:
                if left in intervened.inputs and right in intervened.inputs:
                    supported |= get_modality_validity(intervened, left) & get_modality_validity(intervened, right)

        return None if bool(supported.all()) else "no configured directed pair remains available"

    def forward(self, batch: Batch) -> ModelOutput:
        feature_map = prepare_projected_features(self, batch, self.names)
        values, valid, pair_attention = self._interactions(feature_map)
        fused, group_weights, group_attention = self._fuse_groups(values, valid)
        fused = self.fusion(fused)

        aux: dict[str, torch.Tensor] = {
            "fused": fused,
            "group_weights": group_weights,
        }
        for key, value in values.items():
            aux[f"pair_{key}"] = value
            aux[f"pair_valid_{key}"] = valid[key]
            aux[f"pair_attention_{key}"] = pair_attention[key]

        aux.update(group_attention)
        return ModelOutput(preds=self.prediction(fused), aux=aux)


@MODELS.register("tfs_fusion_model")
def tfs_fusion_model(**kwargs: object) -> TFSFusionModel:
    return TFSFusionModel(**kwargs)
