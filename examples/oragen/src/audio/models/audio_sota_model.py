import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel

from common.utils import multitask_dict_to_tensor

class ModelHead(BaseModel):
    def __init__(self, config, num_labels) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderSOTAModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.post_init()

    def forward(self, batch: Batch) -> ModelOutput:
        outputs = self.wav2vec2(batch.inputs["audio"])
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = self.gender(hidden_states)
        outputs = {'gen': logits_gender[:, 0:2], 'age': logits_age[:, 0]}
        return ModelOutput(preds=multitask_dict_to_tensor(outputs), aux=outputs)


@MODELS.register("agender_sota_model")
def agender_sota_model(**params):
    return AgeGenderSOTAModel(**params)
