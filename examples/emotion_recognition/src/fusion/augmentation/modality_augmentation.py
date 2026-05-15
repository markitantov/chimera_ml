import random

import torch


class ModalityDropAugmentation(torch.nn.Module):
    def __init__(
        self,
        *,
        keep_prob: float = 0.4,
        drop_audio_prob: float = 0.1,
        drop_video_prob: float = 0.1,
        drop_text_prob: float = 0.1,
        drop_audio_video_prob: float = 0.1,
        drop_audio_text_prob: float = 0.1,
        drop_video_text_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.keep_prob = float(keep_prob)
        self.drop_audio_prob = float(drop_audio_prob)
        self.drop_video_prob = float(drop_video_prob)
        self.drop_text_prob = float(drop_text_prob)
        self.drop_audio_video_prob = float(drop_audio_video_prob)
        self.drop_audio_text_prob = float(drop_audio_text_prob)
        self.drop_video_text_prob = float(drop_video_text_prob)

        total = (
            self.keep_prob
            + self.drop_audio_prob
            + self.drop_video_prob
            + self.drop_text_prob
            + self.drop_audio_video_prob
            + self.drop_audio_text_prob
            + self.drop_video_text_prob
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ModalityDropAugmentation probabilities must sum to 1.0, got {total:.6f}.")

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        keys = list(inputs.keys())
        draw = random.random()
        boundaries = (
            self.keep_prob,
            self.keep_prob + self.drop_audio_prob,
            self.keep_prob + self.drop_audio_prob + self.drop_video_prob,
            self.keep_prob + self.drop_audio_prob + self.drop_video_prob + self.drop_text_prob,
            self.keep_prob
            + self.drop_audio_prob
            + self.drop_video_prob
            + self.drop_text_prob
            + self.drop_audio_video_prob,
            self.keep_prob
            + self.drop_audio_prob
            + self.drop_video_prob
            + self.drop_text_prob
            + self.drop_audio_video_prob
            + self.drop_audio_text_prob,
        )

        output = dict(inputs)
        if draw < boundaries[0]:
            return output

        if draw < boundaries[1]:
            output[keys[0]] = torch.zeros_like(output[keys[0]])
            return output

        if draw < boundaries[2]:
            output[keys[1]] = torch.zeros_like(output[keys[1]])
            return output

        if draw < boundaries[3]:
            output[keys[2]] = torch.zeros_like(output[keys[2]])
            return output

        if draw < boundaries[4]:
            output[keys[0]] = torch.zeros_like(output[keys[0]])
            output[keys[1]] = torch.zeros_like(output[keys[1]])
            return output

        if draw < boundaries[5]:
            output[keys[0]] = torch.zeros_like(output[keys[0]])
            output[keys[2]] = torch.zeros_like(output[keys[2]])
            return output

        output[keys[1]] = torch.zeros_like(output[keys[1]])
        output[keys[2]] = torch.zeros_like(output[keys[2]])

        return output
