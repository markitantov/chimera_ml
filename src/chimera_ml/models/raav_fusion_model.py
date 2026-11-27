from collections.abc import Sequence

import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel
from chimera_ml.models.common_layers import SafeBatchNorm1d
from chimera_ml.models.utils import (
    build_projections,
    get_modality_mask,
    get_modality_validity,
    normalize_sequence,
    validate_input_dimensions,
)


class RAAVFusionModel(BaseModel):
    """Framewise reliability-aware fusion with contextual refinement."""

    def __init__(
        self,
        input_dims: dict[str, int],
        shared_dim: int,
        output_dim: int,
        frame_modalities: Sequence[str],
        context_modalities: Sequence[str],
        num_context_latents: int = 4,
        gate_hidden: int | None = None,
        dropout: float = 0.1,
        num_heads: int = 4,
        **_: object,
    ):
        super().__init__()
        self.input_dims = validate_input_dimensions(input_dims)
        self.frame_modalities = tuple(frame_modalities)
        self.context_modalities = tuple(context_modalities)

        if not self.frame_modalities:
            raise ValueError("frame_modalities must not be empty.")

        if not self.context_modalities:
            raise ValueError("context_modalities must not be empty.")

        names = (*self.frame_modalities, *self.context_modalities)
        if any(name not in self.input_dims for name in names):
            raise ValueError("All configured modalities must be present in input_dims.")

        if len(set(names)) != len(names):
            raise ValueError("frame_modalities and context_modalities must not overlap.")

        if shared_dim <= 0:
            raise ValueError(f"shared_dim must be positive, got {shared_dim}.")

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")

        if num_context_latents <= 0:
            raise ValueError(f"num_context_latents must be positive, got {num_context_latents}.")

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")

        gate_hidden = shared_dim if gate_hidden is None else gate_hidden
        if gate_hidden <= 0:
            raise ValueError(f"gate_hidden must be positive, got {gate_hidden}.")

        self.names = tuple(dict.fromkeys(names))
        self.proj = build_projections(self.names, self.input_dims, shared_dim)
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(shared_dim, gate_hidden),
                    nn.Tanh(),
                    nn.Linear(gate_hidden, 1),
                )
                for name in self.frame_modalities
            }
        )
        heads = min(shared_dim, num_heads)

        while shared_dim % heads:
            heads -= 1

        self.context_latents = nn.ParameterDict(
            {
                name: nn.Parameter(torch.randn(1, num_context_latents, shared_dim) * 0.02)
                for name in self.context_modalities
            }
        )
        self.context_compression = nn.ModuleDict(
            {
                name: nn.MultiheadAttention(
                    shared_dim,
                    heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for name in self.context_modalities
            }
        )
        self.context_compression_norm = nn.ModuleDict(
            {name: nn.LayerNorm(shared_dim) for name in self.context_modalities}
        )
        self.context_attention = nn.ModuleDict(
            {
                name: nn.MultiheadAttention(
                    shared_dim,
                    heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for name in self.context_modalities
            }
        )
        self.context_norm = nn.LayerNorm(shared_dim)
        self.prediction = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            SafeBatchNorm1d(shared_dim),
            nn.Tanh(),
            nn.Linear(shared_dim, output_dim),
        )

    def _frame_features(
        self,
        batch: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        features: list[torch.Tensor] = []
        validities: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        frame_values: dict[str, torch.Tensor] = {}
        time_steps: int | None = None
        batch_size: int | None = None

        for name in self.frame_modalities:
            if name not in batch.inputs:
                continue

            values = normalize_sequence(batch.inputs[name])
            projected = self.proj[name](values)
            valid = get_modality_mask(batch, name, projected)

            if batch_size is not None and values.shape[0] != batch_size:
                raise ValueError(f"Modality '{name}' has an incompatible batch dimension.")

            if time_steps is not None and values.shape[1] != time_steps:
                raise ValueError("Frame modalities must have the same temporal length.")

            batch_size = values.shape[0]
            time_steps = values.shape[1]
            frame_values[name] = projected
            features.append(projected)
            validities.append(valid)
            scores.append(self.gates[name](projected).squeeze(-1).masked_fill(~valid, -1e4))

        if not features:
            raise ValueError("RAAVFusionModel has no available frame modalities.")

        stacked = torch.stack(features, dim=2)
        available = torch.stack(validities, dim=2)
        weights = torch.softmax(torch.stack(scores, dim=2), dim=2)
        weights = weights * available.to(weights.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        frame_valid = available.any(dim=2)
        fused = fused * frame_valid.unsqueeze(-1).to(fused.dtype)

        return fused, frame_valid, weights, frame_values

    def _context_features(
        self,
        batch: Batch,
        reference: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        features: dict[str, torch.Tensor] = {}
        validities: dict[str, torch.Tensor] = {}
        compression_attention: dict[str, torch.Tensor] = {}
        batch_size = reference.shape[0]

        for name in self.context_modalities:
            if name not in batch.inputs:
                continue

            values = normalize_sequence(batch.inputs[name])
            if values.shape[0] != batch_size:
                raise ValueError(f"Modality '{name}' has an incompatible batch dimension.")

            projected = self.proj[name](values)
            valid = get_modality_mask(batch, name, projected)
            has_context = valid.any(dim=1)
            latents = self.context_latents[name].expand(batch_size, -1, -1).clone()
            attention = projected.new_zeros(batch_size, latents.shape[1], projected.shape[1])

            if has_context.any():
                updated, coefficients = self.context_compression[name](
                    latents[has_context],
                    projected[has_context],
                    projected[has_context],
                    key_padding_mask=~valid[has_context],
                    need_weights=True,
                )
                latents[has_context] = self.context_compression_norm[name](latents[has_context] + updated)
                attention[has_context] = coefficients

            compact_valid = has_context[:, None].expand(-1, latents.shape[1])
            features[name] = latents * compact_valid.unsqueeze(-1).to(latents.dtype)
            validities[name] = compact_valid
            compression_attention[name] = attention

        return features, validities, compression_attention

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

        for name in self.frame_modalities:
            if name in intervened.inputs:
                supported |= get_modality_validity(intervened, name)

        return None if bool(supported.all()) else "no usable frame modality remains"

    def forward(self, batch: Batch) -> ModelOutput:
        frame, frame_valid, frame_weights, frame_values = self._frame_features(batch)
        contexts, context_validities, compression_attention = self._context_features(batch, frame)
        refined = frame.clone()
        context_delta = torch.zeros_like(frame)
        aux: dict[str, torch.Tensor] = {
            "frame_weights": frame_weights.permute(0, 2, 1),
            "frame_mask": frame_valid,
        }

        for name in self.context_modalities:
            if name not in contexts:
                continue

            context = contexts[name]
            context_valid = context_validities[name]
            has_context = context_valid.any(dim=1)
            attention = frame.new_zeros(frame.shape[0], frame.shape[1], context.shape[1])

            if has_context.any():
                attended, attention_valid = self.context_attention[name](
                    frame[has_context],
                    context[has_context],
                    context[has_context],
                    key_padding_mask=~context_valid[has_context],
                    need_weights=True,
                )
                attention[has_context] = attention_valid
                context_delta[has_context] += attended

            aux[f"context_{name}"] = context
            aux[f"context_mask_{name}"] = context_valid
            aux[f"context_attention_{name}"] = attention
            aux[f"context_compression_attention_{name}"] = compression_attention[name]

        refined = self.context_norm(frame + context_delta)
        refined = refined * frame_valid.unsqueeze(-1).to(refined.dtype)
        flat = refined.reshape(-1, refined.shape[-1])

        if refined.shape[1] == 0:
            preds = refined.new_zeros(refined.shape[0], 0, self.prediction[-1].out_features)
        else:
            preds = self.prediction(flat).reshape(refined.shape[0], refined.shape[1], -1)
            preds = preds * frame_valid.unsqueeze(-1).to(preds.dtype)

        aux["context_delta"] = context_delta
        aux["fused"] = refined

        for name, values in frame_values.items():
            aux[f"frame_{name}"] = values

        return ModelOutput(preds=preds, aux=aux)


@MODELS.register("raav_fusion_model")
def raav_fusion_model(**kwargs: object) -> RAAVFusionModel:
    return RAAVFusionModel(**kwargs)
