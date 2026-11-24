import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models._fusion_common import _attention, _features, _mean, _projections
from chimera_ml.models.base import BaseModel


class TFSFusionModel(BaseModel):
    """Hierarchical directed pair fusion (TFS)."""

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

        pairs = [(a, b) for a in self.names for b in self.names if a != b]
        self.groups = tuple(tuple(group) for group in groups) if groups else (tuple(pairs),)
        self.group_layers = nn.ModuleList(nn.Linear(shared_dim, shared_dim) for _ in self.groups)
        self.output = nn.Linear(shared_dim, shared_dim)

    def _pair(
        self,
        left: tuple[torch.Tensor, torch.Tensor],
        right: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, query_valid = left
        key, key_valid = right
        return _mean(query + _attention(query, key, key_valid), query_valid), query_valid.any(1) & key_valid.any(1)

    def forward(self, batch: Batch) -> ModelOutput:
        feats = _features(self, batch, self.names)
        values: dict[str, torch.Tensor] = {}
        for group in self.groups:
            for i, j in group:
                if i not in feats or j not in feats:
                    continue
                values[f"{i}->{j}"], _ = self._pair(feats[i], feats[j])

        if not values:
            raise ValueError("TFSFusionModel needs at least two available modalities.")

        group_values = []
        for layer, group in zip(self.group_layers, self.groups, strict=False):
            current = [values[f"{i}->{j}"] for i, j in group if f"{i}->{j}" in values]
            if current:
                group_values.append(layer(torch.stack(current).mean(0)))

        fused = self.output(torch.stack(group_values).mean(0))
        aux = {f"pair_{key}": value for key, value in values.items()}
        aux["fused"] = fused
        return ModelOutput(preds=self.head(fused), aux=aux)


@MODELS.register("tfs_fusion_model")
def tfs_fusion_model(**kwargs) -> TFSFusionModel:
    return TFSFusionModel(**kwargs)
