import random

import torch
from common.utils import DatasetType


class ModalityDropAugmentation(torch.nn.Module):
    def __init__(
        self,
        audio_drop_prob: float = 0.3,
        video_drop_prob: float = 0.3,
        keep_both_prob: float | None = None,
    ) -> None:
        super().__init__()
        self.audio_drop_prob = float(audio_drop_prob)
        self.video_drop_prob = float(video_drop_prob)
        if keep_both_prob is None:
            keep_both_prob = 1.0 - self.audio_drop_prob - self.video_drop_prob
        
        self.keep_both_prob = float(keep_both_prob)

        probs = {
            "audio_drop_prob": self.audio_drop_prob,
            "video_drop_prob": self.video_drop_prob,
            "keep_both_prob": self.keep_both_prob,
        }
        for name, value in probs.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")

        total = self.audio_drop_prob + self.video_drop_prob + self.keep_both_prob
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "ModalityDropAugmentation probabilities must sum to 1.0. "
                f"Got keep_both_prob={self.keep_both_prob}, "
                f"audio_drop_prob={self.audio_drop_prob}, "
                f"video_drop_prob={self.video_drop_prob}."
            )

    def __call__(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        audio, video = inputs
        draw = random.random()
        if draw < self.keep_both_prob:
            return audio, video

        if draw < self.keep_both_prob + self.audio_drop_prob:
            audio = torch.zeros_like(audio)
        else:
            video = torch.zeros_like(video)

        return audio, video


class MultiAugment(torch.nn.Module):
    def __init__(self, dataset_type) -> None:
        super().__init__()
        self.dataset_type = dataset_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_data, v_data = x
        if self.dataset_type == DatasetType.BOTH:
            return a_data, v_data

        if self.dataset_type == DatasetType.AUDIO:
            v_data = torch.zeros(v_data.shape)
        else:
            a_data = torch.zeros(a_data.shape)

        return a_data, v_data
