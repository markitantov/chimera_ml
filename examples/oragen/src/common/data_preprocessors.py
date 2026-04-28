import torch
from transformers import AutoFeatureExtractor, ViTImageProcessor


class Wav2Vec2DataPreprocessor:
    def __init__(
        self,
        preprocessor_name: str = "facebook/wav2vec2-large-robust",
        sr: int = 16000,
        win_max_length: int = 4,
        return_attention_mask: bool = False,
    ) -> None:
        self.sr = sr
        self.win_max_length = win_max_length

        self.return_attention_mask = return_attention_mask
        self.processor = AutoFeatureExtractor.from_pretrained(preprocessor_name)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        data = self.processor(
            x,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding="max_length",
            max_length=self.sr * self.win_max_length,
            return_attention_mask=self.return_attention_mask,
        )

        return data if self.return_attention_mask else data["input_values"][0]


class HuBERTDataPreprocessor:
    def __init__(
        self,
        preprocessor_name: str = "facebook/wav2vec2-large-robust",
        sr: int = 16000,
        win_max_length: int = 4,
        return_attention_mask: bool = False,
    ) -> None:
        self.sr = sr
        self.win_max_length = win_max_length

        self.return_attention_mask = return_attention_mask
        self.processor = AutoFeatureExtractor.from_pretrained(preprocessor_name)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        data = self.processor(
            x,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding="max_length",
            max_length=self.sr * self.win_max_length,
            return_attention_mask=self.return_attention_mask,
        )

        return data if self.return_attention_mask else data["input_values"][0]


class ViTDataPreprocessor:
    def __init__(self, preprocessor_name: str = "nateraw/vit-age-classifier") -> None:
        self.processor = ViTImageProcessor.from_pretrained(preprocessor_name)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        data = self.processor(x, return_tensors="pt")["pixel_values"]
        return data[0]
