import random

import numpy as np
import torch
import torchaudio


class PolarityInversion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return torch.neg(wave)


class WhiteNoise(torch.nn.Module):
    def __init__(self, min_snr: float = 0.0001, max_snr: float = 0.005) -> None:
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        std = torch.std(audio).numpy()
        noise_std = random.uniform(self.min_snr * std, self.max_snr * std)
        noise = np.random.normal(0.0, noise_std, size=audio.shape).astype(np.float32)

        return audio + torch.Tensor(noise)


class SoxEffect(torch.nn.Module):
    def __init__(self, effects: list[list[str]], sr: int = 16000) -> None:
        super().__init__()
        self.effects = effects
        self.sr = sr

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        wave, _sr = torchaudio.sox_effects.apply_effects_tensor(wave, self.sr, self.effects)
        return wave


class Gain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1) -> None:
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        return torchaudio.transforms.Vol(gain, gain_type="db")(wave)


class RandomChoice(torch.nn.Module):
    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = transforms

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        t = random.choice(self.transforms)
        return t(wave)


class ResampleAudio(torch.nn.Module):
    def __init__(self, orig_sr: int = 32000, new_sr: int = 16000) -> None:
        super().__init__()
        if orig_sr != new_sr:
            self.transforms = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)
        else:
            self.transforms = None

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.transforms(wave) if self.transforms else wave
