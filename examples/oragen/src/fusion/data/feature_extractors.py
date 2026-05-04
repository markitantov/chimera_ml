from typing import Any

import torch
from audio.models.audio_models import (
    AGenderAudioHuBERTModel,
    AGenderAudioW2V2Model,
    _reset_custom_buffers,
)
from common.data_preprocessors import (
    HuBERTDataPreprocessor,
    ViTDataPreprocessor,
    Wav2Vec2DataPreprocessor,
)
from common.utils import FeaturesType, define_context_length, read_img
from image.models.image_models import AGenderImageVITDPALModel, AGenderImageVITGSAModel
from transformers import AutoConfig


class AudioFeatureExtractor:
    def __init__(
        self,
        hf_model_name: str,
        checkpoint_path: str,
        features_type: FeaturesType,
        sr: int = 16000,
        win_max_length: int = 4,
        gender_num_classes: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.features_type = FeaturesType(features_type)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        model_config = AutoConfig.from_pretrained(hf_model_name)
        model_config.output_size = gender_num_classes + 1
        model_config.context_length = define_context_length(int(win_max_length))

        if "hubert" in hf_model_name:
            self.model = AGenderAudioHuBERTModel.from_pretrained(hf_model_name, config=model_config)

            self.preprocessor = HuBERTDataPreprocessor(
                preprocessor_name=hf_model_name,
                sr=sr,
                win_max_length=win_max_length,
            )
        else:
            self.model = AGenderAudioW2V2Model.from_pretrained(hf_model_name, config=model_config)

            self.preprocessor = Wav2Vec2DataPreprocessor(
                preprocessor_name=hf_model_name,
                sr=sr,
                win_max_length=win_max_length,
            )

        _reset_custom_buffers(self.model)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self.preprocessor(waveform).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.features_type == FeaturesType.EARLY:
                features = self.model.early_features(waveform)
            elif self.features_type == FeaturesType.INTERMEDIATE:
                features = self.model.intermediate_features(waveform)
            else:
                features = self.model.late_features(waveform)

        return features.detach().cpu().squeeze()


class ImageFeatureExtractor:
    def __init__(
        self,
        hf_model_name: str,
        checkpoint_path: str,
        features_type: FeaturesType,
        win_max_length: int,
        device: str | torch.device | None = None,
    ) -> None:
        self.features_type = FeaturesType(features_type)
        self.win_max_length = win_max_length
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        if "gsa" in checkpoint_path.lower():
            self.model = AGenderImageVITGSAModel()
        else:
            self.model = AGenderImageVITDPALModel()

        self.preprocessor = ViTDataPreprocessor(preprocessor_name=hf_model_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, images_or_paths: list[Any]) -> torch.Tensor:
        images_or_paths = images_or_paths[: self.win_max_length] + (
            [images_or_paths[-1]] * max(0, self.win_max_length - len(images_or_paths))
        )
        images = [self.preprocessor(read_img(item)) for item in images_or_paths]
        batched_images = torch.stack(images, dim=0).to(self.device)

        with torch.no_grad():
            if self.features_type == FeaturesType.EARLY:
                features = self.model.early_features(batched_images)
            elif self.features_type == FeaturesType.INTERMEDIATE:
                features = self.model.intermediate_features(batched_images)
            else:
                features = self.model.late_features(batched_images)

        return features.detach().cpu()
