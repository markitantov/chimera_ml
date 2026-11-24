from __future__ import annotations

import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models._utils import _features, _masked_softmax, _projections
from chimera_ml.models.base import BaseModel


class _CrossAttention(nn.Module):
    """Single-head cross-attention with explicit diagnostics and safe masking."""

    def __init__(self, shared_dim: int):
        super().__init__()
        self.query = nn.Linear(shared_dim, shared_dim)
        self.key = nn.Linear(shared_dim, shared_dim)
        self.value = nn.Linear(shared_dim, shared_dim)
        self.output = nn.Linear(shared_dim, shared_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_projection = self.query(query)
        key_projection = self.key(key)
        value_projection = self.value(key)
        scores = query_projection @ key_projection.transpose(-1, -2) * query_projection.shape[-1] ** -0.5
        weights = _masked_softmax(scores, key_valid)
        result = self.output(weights.to(value_projection.dtype) @ value_projection)
        return result, weights


class RAAVFusionModel(BaseModel):
    """Adaptive frame fusion with bottleneck context and frame-to-context attention."""

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        shared_dim: int,
        head: nn.Module,
        frame_modalities: tuple[str, ...],
        context_modality: str,
        input_dims: dict[str, int] | None = None,
        num_latents: int = 4,
        gate_hidden: int = 64,
        temporal: nn.Module | None = None,
    ):
        super().__init__()
        if num_latents <= 0:
            raise ValueError(f"num_latents must be positive, got {num_latents}.")
        self.names = tuple(encoders)
        self.encoders = nn.ModuleDict(encoders)
        self.shared_dim = shared_dim
        input_dims = input_dims or {}
        self.proj = _projections(self.names, input_dims, shared_dim)
        self.head = head
        self.frame_modalities = tuple(frame_modalities)
        if not self.frame_modalities:
            raise ValueError("RAAVFusionModel requires at least one frame modality.")
        unknown = set(self.frame_modalities) - set(self.names)
        if context_modality not in self.names or unknown:
            raise ValueError("RAAV modalities must be present in encoders.")
        self.context_modality = context_modality
        self.gate = nn.Sequential(nn.Linear(shared_dim, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
        self.latents = nn.Parameter(torch.randn(num_latents, shared_dim) / shared_dim**0.5)
        self.context_latent_attn = _CrossAttention(shared_dim)
        self.frame_to_context_attn = _CrossAttention(shared_dim)
        self.audio_latent_attn = self.context_latent_attn
        self.frame_to_audio_attn = self.frame_to_context_attn
        self.context_latent_norm = nn.LayerNorm(shared_dim)
        self.frame_context_norm = nn.LayerNorm(shared_dim)
        self.temporal = temporal
        self.refine = nn.Linear(shared_dim, shared_dim)

    def _frame_fusion(
        self,
        frames: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = frames[0][0].shape[1]
        if any(value.shape[1] != length for value, _ in frames):
            raise ValueError("RAAV frame modalities must have the same temporal length.")

        values = torch.stack([value for value, _ in frames], dim=1)
        valid = torch.stack([mask for _, mask in frames], dim=1)
        batch_size, modality_count, time_steps, _ = values.shape
        scores = self.gate(values).squeeze(-1).permute(0, 2, 1)
        available = valid.permute(0, 2, 1)
        weights = _masked_softmax(
            scores.reshape(batch_size * time_steps, modality_count), available.reshape(-1, modality_count)
        )
        weights = weights.reshape(batch_size, time_steps, modality_count).permute(0, 2, 1)
        fused = (weights.unsqueeze(-1) * values).sum(1)
        frame_valid = valid.any(1)
        fused = fused * frame_valid.to(fused.dtype).unsqueeze(-1)
        return fused, frame_valid, weights

    def forward(self, batch: Batch) -> ModelOutput:
        feats = _features(self, batch, (*self.frame_modalities, self.context_modality))
        frames = [feats[name] for name in self.frame_modalities if name in feats]
        if not frames:
            raise ValueError("RAAVFusionModel has no available frame modalities.")

        fused, frame_valid, frame_weights = self._frame_fusion(frames)
        context = feats.get(self.context_modality)
        batch_size, frame_length = fused.shape[:2]
        latent_query = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        if context is None:
            context_valid = torch.zeros(batch_size, dtype=torch.bool, device=fused.device)
            context_latents = fused.new_zeros(latent_query.shape)
            latent_weights = fused.new_zeros(batch_size, self.latents.shape[0], 0)
            frame_context = fused.new_zeros(fused.shape)
            frame_context_weights = fused.new_zeros(batch_size, frame_length, self.latents.shape[0])
        else:
            context_features, context_mask = context
            context_valid = context_mask.any(1)
            context_update, latent_weights = self.context_latent_attn(latent_query, context_features, context_mask)
            context_latents = self.context_latent_norm(latent_query + context_update)
            context_latents = context_latents * context_valid.to(context_latents.dtype)[:, None, None]
            latent_valid = context_valid[:, None].expand(-1, self.latents.shape[0])
            frame_context, frame_context_weights = self.frame_to_context_attn(fused, context_latents, latent_valid)
            frame_context = frame_context * context_valid.to(frame_context.dtype)[:, None, None]

        context_delta = self.refine(frame_context) * context_valid.to(frame_context.dtype)[:, None, None]
        refined = self.frame_context_norm(fused + context_delta)
        refined = refined * frame_valid.to(refined.dtype).unsqueeze(-1)
        if self.temporal is not None:
            if isinstance(self.temporal, nn.TransformerEncoder):
                temporal_out = refined.new_zeros(refined.shape)
                rows_with_frames = frame_valid.any(1)
                if rows_with_frames.any():
                    temporal_out[rows_with_frames] = self.temporal(
                        refined[rows_with_frames], src_key_padding_mask=~frame_valid[rows_with_frames]
                    )
                refined = temporal_out
            else:
                refined = self.temporal(refined)
            if refined.ndim != 3 or refined.shape[:2] != frame_valid.shape or refined.shape[-1] != self.shared_dim:
                raise ValueError("RAAV temporal module must preserve shape [B, T, shared_dim].")
            refined = refined * frame_valid.to(refined.dtype).unsqueeze(-1)

        preds = self.head(refined)
        if preds.ndim >= 2 and preds.shape[:2] == frame_valid.shape:
            preds = preds * frame_valid.to(preds.dtype).unsqueeze(-1)
        aux = {
            "frame_weights": frame_weights,
            "frame_mask": frame_valid,
            "fused": fused,
            "context": context_latents,
            "context_mask": context_valid,
            "context_latent_attention": latent_weights,
            "frame_to_context_attention": frame_context_weights,
            "context_delta": context_delta,
            "refined": refined,
        }
        return ModelOutput(preds=preds, aux=aux)


@MODELS.register("raav_fusion_model")
def raav_fusion_model(**kwargs) -> RAAVFusionModel:
    return RAAVFusionModel(**kwargs)
