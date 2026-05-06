import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torchaudio

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference import InferenceContext


def _decode_with_torchcodec(*, input_path: Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(
        input_path,
        sample_rate=sample_rate,
    )
    samples = decoder.get_all_samples()
    return samples.data, int(samples.sample_rate)


def _decode_with_ffmpeg(
    *,
    input_path: Path,
    work_dir: Path,
    sample_rate: int,
    mono: bool,
    codec: str,
) -> tuple[torch.Tensor, int]:
    output_path = work_dir / f"{input_path.stem}_audio.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-async",
        "1",
        "-vn",
        "-acodec",
        codec,
        "-ar",
        str(sample_rate),
    ]

    if mono:
        command.extend(["-ac", "1"])

    command.append(str(output_path))
    subprocess.run(command, check=True, capture_output=True, text=True)

    try:
        waveform, actual_sample_rate = torchaudio.load(str(output_path))
    finally:
        output_path.unlink(missing_ok=True)

    return waveform, int(actual_sample_rate)


@dataclass
class ExtractAudioStep:
    backend: str = "auto"
    sample_rate: int = 16000
    mono: bool = True
    codec: str = "pcm_s16le"

    def run(self, ctx: InferenceContext) -> InferenceContext:
        decoder_errors: list[str] = []

        try:
            if self.backend in {"auto", "torchcodec"}:
                waveform, sample_rate = _decode_with_torchcodec(
                    input_path=ctx.input_path,
                    sample_rate=self.sample_rate,
                )
            else:
                raise RuntimeError("torchcodec backend is disabled")
        except Exception as exc:
            decoder_errors.append(f"torchcodec: {exc}")
            if self.backend == "torchcodec":
                raise ValueError(f"Unable to decode audio from media file: {ctx.input_path}") from exc

            try:
                waveform, sample_rate = _decode_with_ffmpeg(
                    input_path=ctx.input_path,
                    work_dir=ctx.work_dir,
                    sample_rate=self.sample_rate,
                    mono=self.mono,
                    codec=self.codec,
                )
            except Exception as ffmpeg_exc:
                decoder_errors.append(f"ffmpeg: {ffmpeg_exc}")
                raise ValueError(
                    "Unable to decode audio from media file "
                    f"{ctx.input_path}. Tried backends: {', '.join(decoder_errors)}"
                ) from ffmpeg_exc

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        audio_waveform = waveform.squeeze(0).contiguous()
        ctx.set_artifact("audio_waveform", audio_waveform)
        ctx.set_artifact("audio_num_samples", int(audio_waveform.numel()))
        ctx.set_artifact("audio_sample_rate", int(sample_rate))
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
