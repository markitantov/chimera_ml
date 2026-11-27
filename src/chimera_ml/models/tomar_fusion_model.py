import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel
from chimera_ml.models.utils import build_projections, get_modality_mask, normalize_sequence, validate_input_dimensions


class TOMARFusionModel(BaseModel):
    """Text-oriented residual fusion with a configurable anchor modality."""

    def __init__(
        self,
        input_dims: dict[str, int],
        shared_dim: int,
        anchor_modality: str,
        output_dim: int,
        gate_hidden: int | None = None,
    ):
        super().__init__()
        self.input_dims = validate_input_dimensions(input_dims)

        if anchor_modality not in self.input_dims:
            raise ValueError(f"Unknown anchor modality '{anchor_modality}'.")

        if shared_dim <= 0:
            raise ValueError(f"shared_dim must be positive, got {shared_dim}.")

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")

        gate_hidden = shared_dim if gate_hidden is None else gate_hidden
        if gate_hidden <= 0:
            raise ValueError(f"gate_hidden must be positive, got {gate_hidden}.")

        self.anchor_modality = anchor_modality
        self.names = tuple(self.input_dims)
        self.extra_modalities = tuple(name for name in self.names if name != anchor_modality)
        self.proj = build_projections(self.names, self.input_dims, shared_dim)
        self.delta = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(shared_dim, gate_hidden),
                    nn.Tanh(),
                    nn.Linear(gate_hidden, shared_dim),
                )
                for name in self.extra_modalities
            }
        )
        self.rho = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(shared_dim * 2, gate_hidden),
                    nn.Tanh(),
                    nn.Linear(gate_hidden, 1),
                )
                for name in self.extra_modalities
            }
        )
        self.norm = nn.LayerNorm(shared_dim)
        self.prediction = nn.Linear(shared_dim, output_dim)

    def forward(self, batch: Batch) -> ModelOutput:
        if self.anchor_modality not in batch.inputs:
            raise ValueError(f"Anchor modality '{self.anchor_modality}' is missing.")

        anchor_input = normalize_sequence(batch.inputs[self.anchor_modality])
        if anchor_input.shape[-1] != self.input_dims[self.anchor_modality]:
            raise ValueError(
                f"Feature '{self.anchor_modality}' has width {anchor_input.shape[-1]}, "
                f"expected {self.input_dims[self.anchor_modality]}."
            )

        anchor_projected = self.proj[self.anchor_modality](anchor_input)
        anchor_validity = get_modality_mask(batch, self.anchor_modality, anchor_projected)
        anchor_available = anchor_validity.any(dim=1)

        if not bool(anchor_available.all()):
            raise ValueError(f"Anchor modality '{self.anchor_modality}' is unavailable for one or more samples.")

        anchor_weight = anchor_validity.to(anchor_projected.dtype).unsqueeze(-1)
        anchor_count = anchor_weight.sum(dim=1).clamp_min(1)
        anchor = (anchor_projected * anchor_weight).sum(dim=1) / anchor_count
        fused = anchor.clone()
        aux: dict[str, torch.Tensor] = {"anchor": anchor}

        for name in self.extra_modalities:
            if name not in batch.inputs:
                continue

            values = normalize_sequence(batch.inputs[name])
            if values.shape[0] != anchor_input.shape[0]:
                raise ValueError(f"Modality '{name}' has an incompatible batch dimension.")

            if values.shape[-1] != self.input_dims[name]:
                raise ValueError(f"Feature '{name}' has width {values.shape[-1]}, expected {self.input_dims[name]}.")

            projected = self.proj[name](values)
            validity = get_modality_mask(batch, name, projected)
            weight = validity.to(projected.dtype).unsqueeze(-1)
            count = weight.sum(dim=1).clamp_min(1)
            representation = (projected * weight).sum(dim=1) / count
            available = validity.any(dim=1)
            delta = self.delta[name](representation) * available.unsqueeze(-1).to(representation.dtype)
            rho = torch.sigmoid(self.rho[name](torch.cat((anchor, representation), dim=-1)))
            rho = rho * available.unsqueeze(-1).to(rho.dtype)
            fused = fused + rho * delta
            aux[f"delta_{name}"] = delta
            aux[f"rho_{name}"] = rho

        fused = self.norm(fused)
        aux["fused"] = fused
        return ModelOutput(preds=self.prediction(fused), aux=aux)


@MODELS.register("tomar_fusion_model")
def tomar_fusion_model(**kwargs: object) -> TOMARFusionModel:
    return TOMARFusionModel(**kwargs)
