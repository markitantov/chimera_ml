import torch
import torch.nn as nn
import torch.nn.functional as F

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import InterventionValidation, ModelOutput
from chimera_ml.models._utils import (
    _attention,
    _features,
    _masked_softmax,
    _mean,
    _modality_validity,
    _projections,
    _validate_input_dims,
)
from chimera_ml.models.base import BaseModel


class _SafeBatchNorm1d(nn.BatchNorm1d):
    """Keep the publication's batch normalization defined for one-sample batches."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.shape[0] == 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(input)


class _PositionalEncoding(nn.Module):
    """MASAI-compatible sinusoidal positional encoding with dropout."""

    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds positional encoding max_len {self.encoding.shape[1]}."
            )
        return self.dropout(x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype, device=x.device))


class _TransformerCrossAttention(nn.Module):
    """A masked, independent Transformer cross-attention block."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        heads = min(8, dim)
        while dim % heads:
            heads -= 1

        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.positional_encoding = _PositionalEncoding(dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.feed_forward_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_valid: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> torch.Tensor:
        if query.shape[1] == 0 or key.shape[1] == 0:
            return torch.zeros_like(query)

        query = self.positional_encoding(query)
        key = self.positional_encoding(key)
        safe_key_valid = key_valid.clone()
        empty_keys = ~safe_key_valid.any(dim=1)
        if empty_keys.any():
            safe_key_valid[empty_keys, 0] = True

        safe_key = key.masked_fill(~key_valid.unsqueeze(-1), 0)
        attended, _ = self.attention(
            query,
            safe_key,
            safe_key,
            key_padding_mask=~safe_key_valid,
            need_weights=False,
        )
        result = self.layer_norm(query + self.dropout(attended))
        result = self.feed_forward_norm(result + self.dropout(self.feed_forward(result)))
        return result * query_valid.unsqueeze(-1).to(result.dtype)


def _stat_pool(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return masked mean and population standard deviation concatenated by width."""
    weight = valid.to(x.dtype).unsqueeze(-1)
    count = weight.sum(dim=1)
    mean = (x * weight).sum(dim=1) / count.clamp_min(1)
    centered = (x - mean.unsqueeze(1)) * weight
    variance = centered.square().sum(dim=1) / count.clamp_min(1)
    positive_variance = variance > 0
    safe_variance = torch.where(positive_variance, variance, torch.ones_like(variance))
    std = safe_variance.sqrt() * positive_variance.to(x.dtype)
    return torch.cat((mean, std), dim=-1)


def _align_and_concatenate(
    values: list[torch.Tensor], validities: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad query streams to a common length before concatenating their channels."""
    length = max(value.shape[1] for value in values)
    padded_values = []
    padded_validities = []
    for value, valid in zip(values, validities, strict=True):
        padding = length - value.shape[1]
        if padding:
            value = F.pad(value, (0, 0, 0, padding))
            valid = F.pad(valid, (0, padding))

        padded_values.append(value)
        padded_validities.append(valid)

    return torch.cat(padded_values, dim=-1), torch.stack(padded_validities, dim=-1).any(dim=-1)


class _TripleFusionBackbone(nn.Module):
    """Encoder-independent MASAI TripleFusion backbone for three modalities."""

    def __init__(self, names: tuple[str, str, str], shared_dim: int):
        super().__init__()
        self.names = names
        self.shared_dim = shared_dim
        self._block_indices = {}
        blocks = []
        for source in names:
            for target in names:
                if source == target:
                    continue

                self._block_indices[(source, target)] = len(blocks)
                blocks.append(_TransformerCrossAttention(shared_dim))

        self.blocks = nn.ModuleList(blocks)
        self.cross_blocks = nn.ModuleList(
            [_TransformerCrossAttention(shared_dim * 3), _TransformerCrossAttention(shared_dim * 3)]
        )
        self.cross_att_dense_layer = nn.Linear(shared_dim * 12, shared_dim)
        self.cross_att_batch_norm = _SafeBatchNorm1d(shared_dim)
        self.cross_att_activation = nn.Tanh()

    def _block(self, source: str, target: str) -> _TransformerCrossAttention:
        return self.blocks[self._block_indices[(source, target)]]

    def forward(
        self,
        features: dict[str, tuple[torch.Tensor, torch.Tensor]],
        gate: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], tuple[str, ...], torch.Tensor]:
        first_triple = ((self.names[0], self.names[1]), (self.names[1], self.names[2]), (self.names[2], self.names[0]))
        second_triple = ((self.names[1], self.names[0]), (self.names[2], self.names[1]), (self.names[0], self.names[2]))
        outputs: dict[str, torch.Tensor] = {}
        output_valid: dict[str, torch.Tensor] = {}
        pair_values: dict[str, torch.Tensor] = {}
        pair_valid: dict[str, torch.Tensor] = {}

        for source, target in first_triple + second_triple:
            source_features, source_valid = features[source]
            target_features, target_valid = features[target]
            key = _pair_name((source, target))
            outputs[key] = self._block(source, target)(
                target_features,
                source_features,
                target_valid,
                source_valid,
            )
            output_valid[key] = target_valid & source_valid.any(dim=1, keepdim=True)
            outputs[key] = outputs[key] * output_valid[key].unsqueeze(-1).to(outputs[key].dtype)
            pair_values[key] = _mean(outputs[key], output_valid[key])
            pair_valid[key] = output_valid[key].any(dim=1)

        pair_keys = tuple(_pair_name(pair) for pair in first_triple + second_triple)
        stacked_values = torch.stack([pair_values[key] for key in pair_keys], dim=1)
        stacked_valid = torch.stack([pair_valid[key] for key in pair_keys], dim=1)
        if gate is None:
            pair_weights = stacked_valid.to(stacked_values.dtype)
        else:
            pair_weights = _masked_softmax(gate(stacked_values).squeeze(-1), stacked_valid)
            for key, value in zip(pair_keys, pair_weights.unbind(dim=1), strict=True):
                outputs[key] = outputs[key] * value[:, None, None]

        avt, avt_valid = _align_and_concatenate(
            [outputs[_pair_name(pair)] for pair in first_triple],
            [output_valid[_pair_name(pair)] for pair in first_triple],
        )
        vta, vta_valid = _align_and_concatenate(
            [outputs[_pair_name(pair)] for pair in second_triple],
            [output_valid[_pair_name(pair)] for pair in second_triple],
        )
        avt_vta = self.cross_blocks[0](vta, avt, vta_valid, avt_valid)
        vta_avt = self.cross_blocks[1](avt, vta, avt_valid, vta_valid)
        fused_sequence, fused_valid = _align_and_concatenate([avt_vta, vta_avt], [vta_valid, avt_valid])
        pooled = _stat_pool(fused_sequence, fused_valid)
        fused = self.cross_att_activation(self.cross_att_batch_norm(self.cross_att_dense_layer(pooled)))
        fused = fused * stacked_valid.any(dim=1, keepdim=True).to(fused.dtype)
        return fused, pair_values, pair_valid, pair_keys, pair_weights


def _pair_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}->{pair[1]}"


def _normalize_groups(
    names: tuple[str, ...], groups: tuple[tuple[tuple[str, str], ...], ...] | None
) -> tuple[tuple[tuple[str, str], ...], ...]:
    if groups is None:
        return tuple(((left, right), (right, left)) for index, left in enumerate(names) for right in names[index + 1 :])

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
    feats: dict[str, tuple[torch.Tensor, torch.Tensor]], groups: tuple[tuple[tuple[str, str], ...], ...], pair_fn
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
    groups: tuple[tuple[tuple[str, str], ...], ...],
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

    With three configured modalities and the default groups this follows the
    MASAI TripleFusion backbone: six independent directed Transformer
    cross-attention blocks, two concatenated triples, reciprocal second-level
    cross-attention, masked mean+STD pooling, and Dense+BatchNorm+Tanh.
    Incomplete or explicitly grouped inputs use the generalized hierarchy.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        shared_dim: int,
        output_dim: int,
        groups: tuple[tuple[tuple[str, str], ...], ...] | None = None,
    ):
        super().__init__()
        self.input_dims = _validate_input_dims(input_dims)
        if shared_dim <= 0:
            raise ValueError(f"shared_dim must be positive, got {shared_dim}.")

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")

        self.names = tuple(self.input_dims)
        self.shared_dim = shared_dim
        self.proj = _projections(self.names, self.input_dims, shared_dim)
        self.prediction = nn.Linear(shared_dim, output_dim)
        self.groups = _normalize_groups(self.names, groups)
        self.group_layers = nn.ModuleList(nn.Linear(shared_dim, shared_dim) for _ in self.groups)
        self.output = nn.Linear(shared_dim, shared_dim)
        self.triple_fusion = _TripleFusionBackbone(self.names, shared_dim) if len(self.names) == 3 else None

    def _pair(
        self,
        left: tuple[torch.Tensor, torch.Tensor],
        right: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _directed_pair(left, right)

    def _uses_triple_branch(self, batch: Batch) -> bool:
        return (
            self.triple_fusion is not None
            and all(name in batch.inputs for name in self.names)
            and self.groups == _normalize_groups(self.names, None)
        )

    def validate_intervention(
        self, original: Batch, intervened: Batch, modality: str, policy: str
    ) -> InterventionValidation:
        if self._uses_triple_branch(original) != self._uses_triple_branch(intervened):
            return InterventionValidation(False, "intervention changes the TFS computational branch")
        size = next(iter(original.inputs.values())).shape[0]
        supported = torch.zeros(size, dtype=torch.bool, device=next(iter(original.inputs.values())).device)
        for group in self.groups:
            for left, right in group:
                if left in intervened.inputs and right in intervened.inputs:
                    supported |= _modality_validity(intervened, left) & _modality_validity(intervened, right)
        return InterventionValidation(supported, "no configured directed pair remains available")

    def forward(self, batch: Batch) -> ModelOutput:
        feats = _features(self, batch, self.names)
        values, valid, _ = _collect_pairs(feats, self.groups, self._pair)
        if not values:
            raise ValueError("TFSFusionModel needs at least two available modalities.")

        if self._uses_triple_branch(batch):
            fused, values, valid, pair_keys, pair_weights = self.triple_fusion(feats)
            group_weights = torch.stack(
                [sum(pair_weights[:, pair_keys.index(_pair_name(pair))] for pair in group) for group in self.groups],
                dim=1,
            )
            group_weights = group_weights / group_weights.sum(dim=1, keepdim=True).clamp_min(1)
            aux: dict[str, torch.Tensor] = {}
            for key, value in values.items():
                aux[f"pair_{key}"] = value
                aux[f"pair_valid_{key}"] = valid[key]

            aux["pair_weights"] = pair_weights
            aux["group_weights"] = group_weights
            aux["fused"] = fused
            return ModelOutput(preds=self.prediction(fused), aux=aux)

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
        return ModelOutput(preds=self.prediction(fused), aux=aux)


class TFSDCMMOEFusionModel(BaseModel):
    """Adaptive hierarchical TFS with input-dependent directed-pair experts."""

    def __init__(
        self,
        input_dims: dict[str, int],
        shared_dim: int,
        output_dim: int,
        groups: tuple[tuple[tuple[str, str], ...], ...] | None = None,
        gate_hidden: int = 64,
    ):
        super().__init__()
        self.input_dims = _validate_input_dims(input_dims)
        if shared_dim <= 0:
            raise ValueError(f"shared_dim must be positive, got {shared_dim}.")

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")

        self.names = tuple(self.input_dims)
        self.shared_dim = shared_dim
        self.proj = _projections(self.names, self.input_dims, shared_dim)
        self.prediction = nn.Linear(shared_dim, output_dim)
        self.groups = _normalize_groups(self.names, groups)
        self.group_layers = nn.ModuleList(nn.Linear(shared_dim, shared_dim) for _ in self.groups)
        self.gates = nn.Sequential(nn.Linear(shared_dim, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
        self.output = nn.Linear(shared_dim, shared_dim)
        self.triple_fusion = _TripleFusionBackbone(self.names, shared_dim) if len(self.names) == 3 else None

    def _pair(
        self,
        left: tuple[torch.Tensor, torch.Tensor],
        right: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _directed_pair(left, right)

    def _uses_triple_branch(self, batch: Batch) -> bool:
        return (
            self.triple_fusion is not None
            and all(name in batch.inputs for name in self.names)
            and self.groups == _normalize_groups(self.names, None)
        )

    def validate_intervention(
        self, original: Batch, intervened: Batch, modality: str, policy: str
    ) -> InterventionValidation:
        if self._uses_triple_branch(original) != self._uses_triple_branch(intervened):
            return InterventionValidation(False, "intervention changes the TFS-DCMMOE computational branch")
        size = next(iter(original.inputs.values())).shape[0]
        supported = torch.zeros(size, dtype=torch.bool, device=next(iter(original.inputs.values())).device)
        for group in self.groups:
            for left, right in group:
                if left in intervened.inputs and right in intervened.inputs:
                    supported |= _modality_validity(intervened, left) & _modality_validity(intervened, right)
        return InterventionValidation(supported, "no configured directed pair remains available")

    def forward(self, batch: Batch) -> ModelOutput:
        feats = _features(self, batch, self.names)
        values, valid, _ = _collect_pairs(feats, self.groups, self._pair)
        if not values:
            raise ValueError("TFSDCMMOEFusionModel needs at least two available modalities.")

        if self._uses_triple_branch(batch):
            fused, values, valid, pair_keys, pair_weights = self.triple_fusion(feats, gate=self.gates)
            group_weights = torch.stack(
                [sum(pair_weights[:, pair_keys.index(_pair_name(pair))] for pair in group) for group in self.groups],
                dim=1,
            )
            group_weights = group_weights / group_weights.sum(dim=1, keepdim=True).clamp_min(1)
            aux: dict[str, torch.Tensor] = {}
            for key, value in values.items():
                aux[f"pair_{key}"] = value
                aux[f"pair_valid_{key}"] = valid[key]

            aux["pair_weights"] = pair_weights
            aux["group_weights"] = group_weights
            aux["fused"] = fused
            return ModelOutput(preds=self.prediction(fused), aux=aux)

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
        return ModelOutput(preds=self.prediction(fused), aux=aux)


@MODELS.register("tfs_fusion_model")
def tfs_fusion_model(**kwargs) -> TFSFusionModel:
    return TFSFusionModel(**kwargs)


@MODELS.register("tfs_dcmmmoe_fusion_model")
def tfs_dcmmmoe_fusion_model(**kwargs) -> TFSDCMMOEFusionModel:
    return TFSDCMMOEFusionModel(**kwargs)
