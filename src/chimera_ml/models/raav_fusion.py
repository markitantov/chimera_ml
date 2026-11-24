import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models._fusion_common import _attention, _features, _projections
from chimera_ml.models.base import BaseModel


class RAAVFusionModel(BaseModel):
    """Asymmetric frame fusion followed by context-latent refinement."""

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
        self.names = tuple(encoders)
        self.encoders = nn.ModuleDict(encoders)
        self.shared_dim = shared_dim
        input_dims = input_dims or {}
        self.proj = _projections(self.names, input_dims, shared_dim)
        self.head = head
        self.frame_modalities = tuple(frame_modalities)
        self.context_modality = context_modality
        self.gate = nn.Sequential(nn.Linear(shared_dim, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
        self.latents = nn.Parameter(torch.randn(num_latents, shared_dim) / shared_dim**0.5)
        self.temporal = temporal
        self.refine = nn.Linear(shared_dim, shared_dim)

    def forward(self, batch: Batch) -> ModelOutput:
        feats = _features(self, batch, (*self.frame_modalities, self.context_modality))
        frames = [feats[name] for name in self.frame_modalities if name in feats]
        if not frames:
            raise ValueError("RAAVFusionModel has no available frame modalities.")

        length = frames[0][0].shape[1]
        if any(value.shape[1] != length for value, _ in frames):
            raise ValueError("RAAV frame modalities must have the same temporal length.")

        x = torch.stack([value for value, _ in frames], 1)
        valid = torch.stack([mask for _, mask in frames], 1)
        scores = self.gate(x).squeeze(-1).masked_fill(~valid, -torch.finfo(x.dtype).max)
        weights = torch.softmax(scores, 1) * valid.to(x.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        fused = (weights.unsqueeze(-1) * x).sum(1)

        context = feats.get(self.context_modality)
        if context is None:
            context_latents = fused.new_zeros((fused.shape[0], self.latents.shape[0], self.shared_dim))
            context_valid = fused.new_zeros(fused.shape[0], dtype=torch.bool)
        else:
            context_features, context_mask = context
            query = self.latents.unsqueeze(0).expand(fused.shape[0], -1, -1)
            context_latents = _attention(query, context_features, context_mask)
            context_valid = context_mask.any(1)

        context_delta = self.refine(context_latents.mean(1))[:, None, :]
        refined = fused + context_delta * context_valid[:, None, None].to(fused.dtype)
        if self.temporal is not None:
            refined = self.temporal(refined)

        aux = {
            "frame_weights": weights,
            "fused": fused,
            "context": context_latents,
            "context_mask": context_valid,
        }
        return ModelOutput(preds=self.head(refined), aux=aux)


@MODELS.register("raav_fusion_model")
def raav_fusion_model(**kwargs) -> RAAVFusionModel:
    return RAAVFusionModel(**kwargs)
