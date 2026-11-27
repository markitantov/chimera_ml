import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models._utils import _features, _mean, _modality_validity, _projections, _validate_input_dims
from chimera_ml.models.base import BaseModel


class TOMARFusionModel(BaseModel):
    """Residual refinement of an anchor modality by additional modalities."""

    def __init__(
        self,
        input_dims: dict[str, int],
        shared_dim: int,
        anchor_modality: str,
        output_dim: int,
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
        if anchor_modality not in self.names:
            raise ValueError(f"Unknown anchor modality: {anchor_modality}")

        self.anchor_modality = anchor_modality
        extras = [name for name in self.names if name != anchor_modality]
        self.rho = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(shared_dim * 2, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
                for name in extras
            }
        )
        self.delta = nn.ModuleDict({name: nn.Linear(shared_dim, shared_dim) for name in extras})
        self.norm = nn.LayerNorm(shared_dim)

    def validate_intervention(self, original: Batch, intervened: Batch, modality: str, policy: str) -> str | None:
        supported = _modality_validity(intervened, self.anchor_modality)
        if modality == self.anchor_modality and policy in {"remove", "mask"}:
            return "anchor modality cannot be removed or masked"
        return None if bool(supported.all()) else "anchor modality is unavailable"

    def forward(self, batch: Batch) -> ModelOutput:
        feats = _features(self, batch, self.names)
        if self.anchor_modality not in feats:
            raise ValueError(f"Anchor modality '{self.anchor_modality}' is missing.")

        anchor_features, anchor_mask = feats[self.anchor_modality]
        anchor_available = anchor_mask.any(1)
        if not anchor_available.all():
            raise ValueError(f"Anchor modality '{self.anchor_modality}' is unavailable for one or more samples.")

        anchor = _mean(anchor_features, anchor_mask)
        correction = torch.zeros_like(anchor)
        aux = {"anchor": anchor}
        for name in self.names:
            if name == self.anchor_modality or name not in feats:
                continue

            value = _mean(*feats[name])
            present = feats[name][1].any(1).to(value.dtype)[:, None]
            joint = torch.cat([anchor, value], dim=-1)
            coefficient = torch.sigmoid(self.rho[name](joint)) * present
            delta = self.delta[name](value) * present
            correction = correction + coefficient * delta
            aux[f"rho_{name}"] = coefficient
            aux[f"delta_{name}"] = delta

        fused = self.norm(anchor + correction)
        aux["fused"] = fused
        return ModelOutput(preds=self.prediction(fused), aux=aux)


@MODELS.register("tomar_fusion_model")
def tomar_fusion_model(**kwargs) -> TOMARFusionModel:
    return TOMARFusionModel(**kwargs)
