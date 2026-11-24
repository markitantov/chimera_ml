import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models._utils import _features, _mean, _projections
from chimera_ml.models.base import BaseModel


class TOMARFusionModel(BaseModel):
    """Residual refinement of an anchor modality by additional modalities."""

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        shared_dim: int,
        head: nn.Module,
        anchor_modality: str,
        input_dims: dict[str, int] | None = None,
        gate_hidden: int = 64,
    ):
        super().__init__()
        self.names = tuple(encoders)
        self.encoders = nn.ModuleDict(encoders)
        self.shared_dim = shared_dim
        input_dims = input_dims or {}
        self.proj = _projections(self.names, input_dims, shared_dim)
        self.head = head
        if anchor_modality not in self.names:
            raise ValueError(f"Unknown anchor modality: {anchor_modality}")
        self.anchor_modality = anchor_modality
        extras = [name for name in self.names if name != anchor_modality]
        self.rho = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(shared_dim, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
                for name in extras
            }
        )
        self.delta = nn.ModuleDict({name: nn.Linear(shared_dim, shared_dim) for name in extras})
        self.norm = nn.LayerNorm(shared_dim)

    def forward(self, batch: Batch) -> ModelOutput:
        feats = _features(self, batch, self.names)
        if self.anchor_modality not in feats:
            raise ValueError(f"Anchor modality '{self.anchor_modality}' is missing.")

        anchor = _mean(*feats[self.anchor_modality])
        correction = torch.zeros_like(anchor)
        aux = {"anchor": anchor}
        for name in self.names:
            if name == self.anchor_modality or name not in feats:
                continue
            value = _mean(*feats[name])
            coefficient = torch.sigmoid(self.rho[name](value))
            delta = self.delta[name](value)
            present = feats[name][1].any(1).to(value.dtype)[:, None]
            correction = correction + coefficient * delta * present
            aux[f"rho_{name}"] = coefficient
            aux[f"delta_{name}"] = delta * present

        fused = self.norm(anchor + correction)
        aux["fused"] = fused
        return ModelOutput(preds=self.head(fused), aux=aux)


@MODELS.register("tomar_fusion_model")
def tomar_fusion_model(**kwargs) -> TOMARFusionModel:
    return TOMARFusionModel(**kwargs)
