from typing import Dict, List

import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.core.registry import MODELS
from chimera_ml.models.base import BaseModel


class GatedPredictionFusionModel(BaseModel):
    """Gated fusion over per-modality logits."""

    def __init__(
        self,
        submodels: Dict[str, BaseModel],
        num_classes: int,
        gate_hidden: int = 64,
        dropout: float = 0.0,
        use_mask: bool = True,
    ):
        super().__init__()
        self.submodels = nn.ModuleDict(submodels)
        self.use_mask = use_mask
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.gate_net = nn.Sequential(
            nn.Linear(num_classes, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )

        self.num_classes = num_classes

    def forward(self, batch: Batch) -> ModelOutput:
        mask_dict = batch.get_masks() if self.use_mask else None

        logits_list = []
        score_list = []
        aux = {}

        for m, model in self.submodels.items():
            if m not in batch.inputs:
                continue

            sub_batch = Batch(inputs={m: batch.inputs[m]}, targets=batch.targets, meta=batch.meta)
            out = model(sub_batch)
            logits = out.preds

            if logits.ndim != 2 or logits.shape[-1] != self.num_classes:
                raise ValueError(f"Submodel '{m}' must output (B, {self.num_classes}) logits.")

            logits = self.dropout(logits)
            score = self.gate_net(logits)

            if mask_dict is not None and m in mask_dict:
                mm = mask_dict[m].to(logits.device).view(-1, 1)
                score = score + (mm - 1.0) * 1e9

            aux[f"logits_{m}"] = logits
            logits_list.append(logits)
            score_list.append(score)

        if not logits_list:
            raise ValueError("No modalities provided in batch.inputs for GatedPredictionFusionModel.")

        L = torch.stack(logits_list, dim=0)  # (M, B, C)
        S = torch.stack(score_list, dim=0)   # (M, B, 1)
        W = torch.softmax(S, dim=0)          # (M, B, 1)

        fused = (W * L).sum(dim=0)           # (B, C)
        aux["gates"] = W.squeeze(-1).transpose(0, 1)  # (B, M)

        return ModelOutput(preds=fused, aux=aux)


@MODELS.register("gated_prediction_fusion_model")
def gated_prediction_fusion_model(
    *,
    submodels: Dict[str, BaseModel],
    num_classes: int | None = None,
    gate_hidden: int = 64,
    dropout: float = 0.0,
    use_mask: bool = True,
    context: dict | None = None,
):
    """Notes
    -----
    ``num_classes`` can be passed explicitly via params, or inferred from
    ``context["num_classes"]`` if a builder provides it.
    """

    if num_classes is None:
        if context is None or "num_classes" not in context:
            raise ValueError("gated_prediction_fusion requires 'num_classes' param or context['num_classes']")
        num_classes = int(context["num_classes"])

    return GatedPredictionFusionModel(
        submodels=submodels,
        num_classes=int(num_classes),
        gate_hidden=gate_hidden,
        dropout=dropout,
        use_mask=use_mask,
    )