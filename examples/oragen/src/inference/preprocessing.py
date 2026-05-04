from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import subprocess

import torch
import torchaudio

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference import InferenceContext


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


@dataclass
class ExtractAudioStep:
    sample_rate: int = 16000
    mono: bool = True
    codec: str = "pcm_s16le"

    def run(self, ctx: InferenceContext) -> InferenceContext:
        output_path = ctx.work_dir / f"{ctx.input_path.stem}_audio.wav"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(ctx.input_path),
            "-async",
            "1",
            "-vn",
            "-acodec",
            self.codec,
            "-ar",
            str(self.sample_rate),
        ]

        if self.mono:
            command.extend(["-ac", "1"])
        
        command.append(str(output_path))
        _run_command(command)

        waveform, sample_rate = torchaudio.load(str(output_path))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        audio_waveform = waveform.squeeze(0).contiguous()
        ctx.set_artifact("audio_waveform", audio_waveform)
        ctx.set_artifact("audio_num_samples", int(audio_waveform.numel()))
        ctx.set_artifact("audio_sample_rate", int(sample_rate))
        output_path.unlink(missing_ok=True)
        return ctx

@dataclass
class VadStep:
    repo_or_dir: str | dict[str, Any] = "snakers4/silero-vad"
    model_name: str = "silero_vad"
    sample_rate: int = 16000
    force_reload: bool = False
    onnx: bool = False
    _model: Any = field(default=None, init=False, repr=False)
    _utils: Any = field(default=None, init=False, repr=False)

    def run(self, ctx: InferenceContext) -> InferenceContext:
        waveform = ctx.get_artifact("audio_waveform")
        self.sample_rate = ctx.get_artifact("audio_sample_rate")
        ctx.set_artifact("vad_segments", self._detect(waveform))
        return ctx

    def _detect(self, waveform: torch.Tensor) -> list[dict[str, int]]:
        model, utils = self._load_model()
        get_speech_timestamps, _, _, _, _ = utils
        timestamps = get_speech_timestamps(
            waveform,
            model,
            sampling_rate=self.sample_rate,
        )

        return [{"start": int(item["start"]), "end": int(item["end"])} for item in timestamps]

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._utils is not None:
            return self._model, self._utils

        self._model, self._utils = torch.hub.load(
            repo_or_dir=str(self.repo_or_dir),
            model=self.model_name,
            force_reload=self.force_reload,
            onnx=self.onnx,
        )

        return self._model, self._utils


@INFERENCE_STEPS.register("extract_audio_step")
def extract_audio_step(**params: Any) -> ExtractAudioStep:
    return ExtractAudioStep(**params)


@INFERENCE_STEPS.register("vad_step")
def vad_step(**params: Any) -> VadStep:
    return VadStep(**params)

