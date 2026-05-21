from typing import Any

import torch
import torch.nn as nn
from fusion.models.common_models import (
    ClassificationHead,
    FeaturesDownsampler,
    PredictionsFusion,
    PredictionsUpsampler,
    TransformerEncoderLayer,
)

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel


class LabelEncoderMeanSimpleFusion(BaseModel):
    def __init__(
        self,
        a_size: tuple[int, ...],
        v_size: tuple[int, ...],
        t_size: tuple[int, ...],
        out_emo: int = 7,
        out_sen: int = 3,
    ) -> None:
        super().__init__()
        self.out_emo = out_emo
        self.out_sen = out_sen
        self.in_t_features = 30
        self.in_f_features = 256
        self.a_downsampler = FeaturesDownsampler(*a_size, self.in_t_features, self.in_f_features)
        self.v_downsampler = FeaturesDownsampler(*v_size, self.in_t_features, self.in_f_features)
        self.t_downsampler = FeaturesDownsampler(*t_size, self.in_t_features, self.in_f_features)

        self.a_upsampler = PredictionsUpsampler(
            *a_size, a_size[0], self.in_f_features, out_emo, out_sen, return_predictions=True
        )
        self.v_upsampler = PredictionsUpsampler(
            *v_size, v_size[0], self.in_f_features, out_emo, out_sen, return_predictions=True
        )
        self.t_upsampler = PredictionsUpsampler(
            *t_size, t_size[0], self.in_f_features, out_emo, out_sen, return_predictions=True
        )

        self.block_a_v = TransformerEncoderLayer(
            input_dim=self.in_f_features, num_heads=4, dropout=0.1, positional_encoding=False
        )
        self.block_a_t = TransformerEncoderLayer(
            input_dim=self.in_f_features, num_heads=4, dropout=0.1, positional_encoding=False
        )

        self.block_v_a = TransformerEncoderLayer(
            input_dim=self.in_f_features, num_heads=4, dropout=0.1, positional_encoding=False
        )
        self.block_v_t = TransformerEncoderLayer(
            input_dim=self.in_f_features, num_heads=4, dropout=0.1, positional_encoding=False
        )

        self.block_t_a = TransformerEncoderLayer(
            input_dim=self.in_f_features, num_heads=4, dropout=0.1, positional_encoding=False
        )
        self.block_t_v = TransformerEncoderLayer(
            input_dim=self.in_f_features, num_heads=4, dropout=0.1, positional_encoding=False
        )

        self.fc1 = nn.Linear(1536, 512)
        self.relu = nn.ReLU()
        self.dp1 = nn.Dropout(p=0.1)
        self.classifier = ClassificationHead(512, out_emo=out_emo, out_sen=out_sen)
        self.predictions_fusion = PredictionsFusion(out_emo=out_emo, out_sen=out_sen, fusion_type="mean")

    def forward(self, batch: Batch) -> ModelOutput:
        a_in, v_in, t_in = batch.inputs["a_features"], batch.inputs["v_features"], batch.inputs["t_features"]
        a = self.a_downsampler(a_in)
        v = self.v_downsampler(v_in)
        t = self.t_downsampler(t_in)

        a_f, ap_predicts = self.a_upsampler(a_in)
        v_f, vp_predicts = self.v_upsampler(v_in)
        t_f, tp_predicts = self.t_upsampler(t_in)

        # a - main feature; v, t - supportive features
        a_v = self.block_a_v(query=v + v_f, key=a, value=a + a_f)
        a_t = self.block_a_t(query=t + t_f, key=a, value=a + a_f)

        # v - main feature; a, t - supportive features
        v_a = self.block_v_a(query=a + a_f, key=v, value=v + v_f)
        v_t = self.block_v_t(query=t + t_f, key=v, value=v + v_f)

        # t - main feature; a, v - supportive features
        t_a = self.block_t_a(query=a + a_f, key=t, value=t + t_f)
        t_v = self.block_t_v(query=v + v_f, key=t, value=t + t_f)

        a_v_t_v = torch.cat((a_v, t_v), dim=-1).mean(dim=1)
        a_t_v_t = torch.cat((a_t, v_t), dim=-1).mean(dim=1)
        v_a_t_a = torch.cat((v_a, t_a), dim=-1).mean(dim=1)

        feats = torch.cat((a_v_t_v, a_t_v_t, v_a_t_a), dim=-1)
        feats = self.dp1(self.relu(self.fc1(feats)))
        feats = feats + a_v_t_v + a_t_v_t + v_a_t_a

        output = self.classifier(feats)
        res = self.predictions_fusion((ap_predicts, vp_predicts, tp_predicts, torch.cat(list(output.values()), dim=-1)))

        return ModelOutput(preds=res, aux={"emo": res[:, 0 : self.out_emo], "sen": res[:, -self.out_sen :]})


@MODELS.register("lefsa_model")
def lefsa_model(context: Any | None = None, **params: Any) -> LabelEncoderMeanSimpleFusion:
    a_size = params.pop("a_size", None)
    if a_size is None and context is not None:
        a_size = context.get("data.a_size", (149, 1024))

    v_size = params.pop("v_size", None)
    if v_size is None and context is not None:
        v_size = context.get("data.v_size", (30, 512))

    t_size = params.pop("t_size", None)
    if t_size is None and context is not None:
        t_size = context.get("data.t_size", (48, 1024))

    return LabelEncoderMeanSimpleFusion(a_size=a_size, v_size=v_size, t_size=t_size, **params)
