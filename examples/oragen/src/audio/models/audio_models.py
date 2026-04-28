import torch
import torch.nn as nn
from common.models import AGenderClassificationHead, PositionalEncoding, StatPoolLayer, TransformerLayer
from common.utils import define_context_length, multitask_dict_to_tensor
from transformers import AutoConfig
from transformers.models.hubert.modeling_hubert import HubertModel, HubertPreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel


class AGenderAudioW2V2Model(Wav2Vec2PreTrainedModel, BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.f_size = int(getattr(config, "hidden_size", 1024))
        self.transformer_block1 = TransformerLayer(
            input_dim=self.f_size,
            num_heads=4,
            dropout=0.1,
            positional_encoding=True,
        )
        self.transformer_block2 = TransformerLayer(
            input_dim=self.f_size,
            num_heads=4,
            dropout=0.1,
            positional_encoding=True,
        )
        self.stp = StatPoolLayer(dim=1)
        self.fc1 = nn.Linear(self.f_size * 2, 256)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=0.6)
        self.cl_head = AGenderClassificationHead(input_size=256, output_size=int(config.output_size))
        self.post_init()
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self) -> None:
        for param in self.wav2vec2.feature_extractor.conv_layers.parameters():
            param.requires_grad = False

    def early_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.wav2vec2.feature_extractor(x)
    
    def intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.wav2vec2(x)[0]

    def late_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wav2vec2(x)[0]
        x = self.relu(self.transformer_block1(x, x, x))
        x = self.relu(self.transformer_block2(x, x, x))
        x = self.stp(x)
        return self.dp(self.relu(self.fc1(x)))
    
    def forward(self, batch: Batch) -> ModelOutput:
        lf = self.late_features(batch.inputs["audio"])
        outputs = self.cl_head(lf)
        return ModelOutput(preds=multitask_dict_to_tensor(outputs), aux=outputs)


class AGenderAudioHuBERTModel(HubertPreTrainedModel, BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.f_size = int(getattr(config, "hidden_size", 1024))
        self.transformer_block1 = TransformerLayer(
            input_dim=self.f_size,
            num_heads=4,
            dropout=0.1,
            positional_encoding=True,
        )
        self.transformer_block2 = TransformerLayer(
            input_dim=self.f_size,
            num_heads=4,
            dropout=0.1,
            positional_encoding=True,
        )
        self.stp = StatPoolLayer(dim=1)
        self.fc1 = nn.Linear(self.f_size * 2, 256)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=0.6)
        self.cl_head = AGenderClassificationHead(input_size=256, output_size=int(config.output_size))
        self.post_init()
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self) -> None:
        for param in self.hubert.feature_extractor.conv_layers.parameters():
            param.requires_grad = False

    def early_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.hubert.feature_extractor(x)

    def intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.hubert(x)[0]

    def late_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hubert(x)[0]
        x = self.relu(self.transformer_block1(x, x, x))
        x = self.relu(self.transformer_block2(x, x, x))
        x = self.stp(x)
        return self.dp(self.relu(self.fc1(x)))

    def forward(self, batch: Batch) -> ModelOutput:
        lf = self.late_features(batch.inputs["audio"])
        outputs = self.cl_head(lf)
        return ModelOutput(preds=multitask_dict_to_tensor(outputs), aux=outputs)


def build_audio_model_config(
    *,
    model_name: str,
    context = None,
):
    win_max_length = context.get("data.win_max_length")
    gender_class_names = context.get("data.gender_class_names")
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.output_size = len(gender_class_names) + 1
    model_config.context_length = define_context_length(int(win_max_length))
    return model_config


def _reset_custom_buffers(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, PositionalEncoding):
            module.reset_parameters()


@MODELS.register("agender_audio_w2v2_model")
def agender_audio_w2v2_model(
    model_name: str,
    context = None,
):
    model_config = build_audio_model_config(
        model_name=model_name,
        context=context,
    )
    model = AGenderAudioW2V2Model.from_pretrained(model_name, config=model_config)
    _reset_custom_buffers(model)
    return model


@MODELS.register("agender_audio_hubert_model")
def agender_audio_hubert_model(
    model_name: str,
    context = None,
):
    model_config = build_audio_model_config(
        model_name=model_name,
        context=context,
    )
    model = AGenderAudioHuBERTModel.from_pretrained(model_name, config=model_config)
    _reset_custom_buffers(model)
    return model
