from typing import Dict, List, Optional

import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.core.registry import MODELS
from chimera_ml.models.base import BaseModel


class FeatureFusionModel(BaseModel):
    """Feature-level fusion with optional masking."""

    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        head: nn.Module,
        dropout: float = 0.0,
        use_mask: bool = True,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = head
        self.use_mask = use_mask

    def forward(self, batch: Batch) -> ModelOutput:
        aux: Dict[str, torch.Tensor] = {}
        embeddings: List[torch.Tensor] = []

        mask_dict = batch.get_masks() if self.use_mask else None

        for modality, encoder in self.encoders.items():
            if modality not in batch.inputs:
                continue

            emb = encoder(batch.inputs[modality])
            aux[f"emb_{modality}"] = emb

            if mask_dict is not None and modality in mask_dict:
                m = mask_dict[modality].to(emb.device).view(-1, 1)
                emb = emb * m

            embeddings.append(emb)

        if not embeddings:
            raise ValueError("No modalities provided in batch.inputs for FeatureFusionModel.")

        fused = torch.cat(embeddings, dim=-1)
        fused = self.dropout(fused)
        preds = self.head(fused)
        return ModelOutput(preds=preds, aux=aux)
    

@MODELS.register("feature_fusion_model")
def feature_fusion_model(
    *,
    encoders: Dict[str, nn.Module],
    head: nn.Module,
    dropout: float = 0.0,
    use_mask: bool = True,
):
    return FeatureFusionModel(encoders=encoders, head=head, dropout=dropout, use_mask=use_mask)


class PredictionFusionModel(BaseModel):
    """Prediction-level fusion: per-modality models -> combine logits."""

    def __init__(
        self,
        submodels: Dict[str, BaseModel],
        fusion: str = "mean",  # "mean" | "sum" | "weighted"
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.submodels = nn.ModuleDict(submodels)
        self.fusion = fusion
        self.weights = weights or {}

    def forward(self, batch: Batch) -> ModelOutput:
        aux = {}
        preds_list = []

        for modality, model in self.submodels.items():
            if modality not in batch.inputs:
                continue

            sub_batch = Batch(inputs={modality: batch.inputs[modality]}, targets=batch.targets, meta=batch.meta)
            out = model(sub_batch)
            aux[f"preds_{modality}"] = out.preds
            preds_list.append((modality, out.preds))

        if not preds_list:
            raise ValueError("No modalities provided in batch.inputs for PredictionFusionModel.")

        if self.fusion == "mean":
            preds = torch.stack([p for _, p in preds_list], dim=0).mean(dim=0)
        elif self.fusion == "sum":
            preds = torch.stack([p for _, p in preds_list], dim=0).sum(dim=0)
        elif self.fusion == "weighted":
            total = 0.0
            acc = None
            for m, p in preds_list:
                w = float(self.weights.get(m, 1.0))
                total += w
                acc = p * w if acc is None else acc + p * w
            preds = acc / max(total, 1e-12)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion}")

        return ModelOutput(preds=preds, aux=aux)
    

@MODELS.register("prediction_fusion_model")
def prediction_fusion_model(
    *,
    submodels: Dict[str, BaseModel],
    fusion: str = "mean",
    weights: Optional[Dict[str, float]] = None,
):
    return PredictionFusionModel(submodels=submodels, fusion=fusion, weights=weights)
