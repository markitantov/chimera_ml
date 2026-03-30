import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel


class GatedFusionModel(BaseModel):
    """Gated (attention-like) fusion over modality embeddings."""

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        head: nn.Module,
        shared_dim: int,
        gate_hidden: int = 64,
        dropout: float = 0.0,
        use_mask: bool = True,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.use_mask = use_mask

        self.proj = nn.ModuleDict({m: nn.Identity() for m in encoders.keys()})

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.gate_net = nn.Sequential(
            nn.Linear(shared_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )

        self.head = head
        self.shared_dim = shared_dim

    def set_projection(self, modality: str, layer: nn.Module) -> None:
        """Allow user to set projection explicitly to ensure shared_dim compatibility."""
        self.proj[modality] = layer

    def forward(self, batch: Batch) -> ModelOutput:
        embs = []
        scores = []
        aux = {}

        for m, enc in self.encoders.items():
            if m not in batch.inputs:
                continue

            emb = enc(batch.inputs[m])
            emb = self.proj[m](emb)

            if emb.shape[-1] != self.shared_dim:
                raise ValueError(
                    f"Embedding dim mismatch for modality '{m}': got {emb.shape[-1]}, expected {self.shared_dim}. "
                    f"Use set_projection('{m}', nn.Linear(..., {self.shared_dim})) or wrap encoder."
                )

            modality_mask = batch.get_masks(f"{m}_mask") if self.use_mask else None
            if modality_mask is not None:
                mm = modality_mask.to(emb.device).view(-1, 1)
                emb = emb * mm

            score = self.gate_net(self.dropout(emb))  # (B, 1)

            if modality_mask is not None:
                mm = modality_mask.to(emb.device).view(-1, 1)
                score = score + (mm - 1.0) * 1e9

            embs.append(emb)
            scores.append(score)
            aux[f"emb_{m}"] = emb

        if not embs:
            raise ValueError("No modalities provided in batch.inputs for GatedFusionModel.")

        E = torch.stack(embs, dim=0)  # (M, B, D)
        S = torch.stack(scores, dim=0)  # (M, B, 1)

        A = torch.softmax(S, dim=0)  # (M, B, 1)
        fused = (A * E).sum(dim=0)   # (B, D)

        aux["gates"] = A.squeeze(-1).transpose(0, 1)  # (B, M)

        preds = self.head(self.dropout(fused))
        return ModelOutput(preds=preds, aux=aux)
    

@MODELS.register("gated_fusion_model")
def gated_fusion_model(
    *,
    encoders: dict[str, nn.Module],
    head: nn.Module,
    shared_dim: int,
    gate_hidden: int = 64,
    dropout: float = 0.0,
    use_mask: bool = True,
) -> GatedFusionModel:
    """Factory for gated feature-fusion model."""
    return GatedFusionModel(
        encoders=encoders,
        head=head,
        shared_dim=shared_dim,
        gate_hidden=gate_hidden,
        dropout=dropout,
        use_mask=use_mask,
    )
