import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torchaudio
from common.utils import find_intersections, slice_audio

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
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    neg_threshold: float | None = None
    min_silence_at_max_speech: int = 98
    use_max_poss_sil_at_max_speech: bool = True
    visualize_probs: bool = False
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
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            max_speech_duration_s=self.max_speech_duration_s,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            visualize_probs=self.visualize_probs,
            neg_threshold=self.neg_threshold,
            min_silence_at_max_speech=self.min_silence_at_max_speech,
            use_max_poss_sil_at_max_speech=self.use_max_poss_sil_at_max_speech,
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


@dataclass
class BuildSegmentsStep:
    use_vad_boundaries: bool = True
    max_segment_length: float = 12.0
    max_pause_length: float = 0.75
    min_segment_length: float = 1.0

    def run(self, ctx: InferenceContext) -> InferenceContext:
        sample_rate = int(ctx.get_artifact("audio_sample_rate"))
        audio_num_samples = int(ctx.get_artifact("audio_num_samples", 0))
        vad_segments = list(ctx.get_artifact("vad_segments", []))
        segments = self._build_segments(
            sample_rate=sample_rate,
            audio_num_samples=audio_num_samples,
            vad_segments=vad_segments,
        )

        ctx.set_artifact("segments", segments)
        return ctx

    def _build_segments(
        self,
        *,
        sample_rate: int,
        audio_num_samples: int,
        vad_segments: list[dict[str, int]],
    ) -> list[dict[str, Any]]:
        if audio_num_samples <= 0:
            return []

        max_segment_samples = max(round(self.max_segment_length * sample_rate), 1)
        min_segment_samples = max(round(self.min_segment_length * sample_rate), 1)
        max_pause_samples = max(round(self.max_pause_length * sample_rate), 0)

        raw_segments: list[dict[str, int]]
        if not self.use_vad_boundaries or not vad_segments:
            raw_segments = slice_audio(
                start_time=0,
                end_time=audio_num_samples,
                win_max_length=max_segment_samples,
                win_shift=max_segment_samples,
                win_min_length=min_segment_samples,
            )
        else:
            raw_segments = []
            current_start = int(vad_segments[0]["start"])
            current_end = int(vad_segments[0]["end"])

            for speech_segment in vad_segments[1:]:
                next_start = int(speech_segment["start"])
                next_end = int(speech_segment["end"])
                gap = next_start - current_end
                merged_length = next_end - current_start
                if gap <= max_pause_samples and merged_length <= max_segment_samples:
                    current_end = next_end
                    continue

                raw_segments.extend(
                    slice_audio(
                        start_time=current_start,
                        end_time=current_end,
                        win_max_length=max_segment_samples,
                        win_shift=max_segment_samples,
                        win_min_length=min_segment_samples,
                    )
                )
                current_start = next_start
                current_end = next_end

            raw_segments.extend(
                slice_audio(
                    start_time=current_start,
                    end_time=current_end,
                    win_max_length=max_segment_samples,
                    win_shift=max_segment_samples,
                    win_min_length=min_segment_samples,
                )
            )

        prepared_segments: list[dict[str, Any]] = []
        for index, segment in enumerate(raw_segments):
            speech_segments = find_intersections([segment], vad_segments) if vad_segments else [segment]
            speech_duration_samples = sum(int(item["end"]) - int(item["start"]) for item in speech_segments)
            start = int(segment["start"])
            end = int(segment["end"])
            duration_samples = max(end - start, 0)
            prepared_segments.append(
                {
                    "index": index,
                    "start": start,
                    "end": end,
                    "start_sec": float(start) / sample_rate,
                    "end_sec": float(end) / sample_rate,
                    "duration_sec": float(duration_samples) / sample_rate,
                    "speech_segments": speech_segments,
                    "speech_duration_sec": float(speech_duration_samples) / sample_rate,
                    "weight": float(max(speech_duration_samples, duration_samples)) / sample_rate,
                }
            )

        return prepared_segments


@dataclass
class TranscribeAudioStep:
    model_name: str = "openai/whisper-small"
    language: str | None = None
    task: str = "transcribe"
    chunk_length_s: int = 30
    batch_size: int = 8
    max_new_tokens: int = 128
    _pipeline: Any = field(default=None, init=False, repr=False)

    def run(self, ctx: InferenceContext) -> InferenceContext:
        waveform = ctx.get_artifact("audio_waveform")
        sample_rate = int(ctx.get_artifact("audio_sample_rate"))
        ctx.set_artifact("transcription", self._transcribe(waveform, sample_rate, ctx.device).strip())
        return ctx

    def _transcribe(self, waveform: torch.Tensor, sample_rate: int, device: str) -> str:
        generate_kwargs = {
            "task": self.task,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.language:
            generate_kwargs["language"] = self.language

        result = self._load_pipeline(device)(
            {"array": waveform.detach().cpu().numpy(), "sampling_rate": sample_rate},
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        return str(result.get("text", "") if isinstance(result, dict) else result)

    def _load_pipeline(self, device: str) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        from transformers import pipeline

        pipeline_device = -1
        if str(device).startswith("cuda") and torch.cuda.is_available():
            pipeline_device = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=pipeline_device,
        )
        return self._pipeline


@dataclass
class TranscribeSegmentsStep:
    respect_vad: bool = True
    model_name: str = "openai/whisper-small"
    language: str | None = None
    task: str = "transcribe"
    chunk_length_s: int = 30
    batch_size: int = 8
    max_new_tokens: int = 128
    _pipeline: Any = field(default=None, init=False, repr=False)

    def run(self, ctx: InferenceContext) -> InferenceContext:
        sample_rate = int(ctx.get_artifact("audio_sample_rate"))
        waveform = ctx.get_artifact("audio_waveform")
        segments = list(ctx.get_artifact("segments", []))
        prepared_segments: list[dict[str, Any]] = []

        for segment in segments:
            start = int(segment["start"])
            end = int(segment["end"])
            has_speech = bool(segment.get("speech_segments"))
            text = ""
            if end > start and (has_speech or not self.respect_vad):
                text = self._transcribe(waveform[start:end].clone(), sample_rate, ctx.device).strip()

            prepared_segments.append(
                {
                    **segment,
                    "text": text,
                    "has_text": bool(text),
                    "has_speech": has_speech,
                }
            )

        ctx.set_artifact("segments", prepared_segments)
        ctx.set_artifact(
            "transcription",
            " ".join(segment["text"] for segment in prepared_segments if segment["text"]).strip(),
        )
        return ctx

    def _transcribe(self, waveform: torch.Tensor, sample_rate: int, device: str) -> str:
        generate_kwargs = {
            "task": self.task,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.language:
            generate_kwargs["language"] = self.language

        result = self._load_pipeline(device)(
            {"array": waveform.detach().cpu().numpy(), "sampling_rate": sample_rate},
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        return str(result.get("text", "") if isinstance(result, dict) else result)

    def _load_pipeline(self, device: str) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        from transformers import pipeline

        pipeline_device = -1
        if str(device).startswith("cuda") and torch.cuda.is_available():
            pipeline_device = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=pipeline_device,
        )
        return self._pipeline


@INFERENCE_STEPS.register("extract_audio_step")
def extract_audio_step(**params: Any) -> ExtractAudioStep:
    return ExtractAudioStep(**params)


@INFERENCE_STEPS.register("build_segments_step")
def build_segments_step(**params: Any) -> BuildSegmentsStep:
    return BuildSegmentsStep(**params)


@INFERENCE_STEPS.register("transcribe_audio_step")
def transcribe_audio_step(**params: Any) -> TranscribeAudioStep:
    return TranscribeAudioStep(**params)


@INFERENCE_STEPS.register("transcribe_segments_step")
def transcribe_segments_step(**params: Any) -> TranscribeSegmentsStep:
    return TranscribeSegmentsStep(**params)


@INFERENCE_STEPS.register("vad_step")
def vad_step(**params: Any) -> VadStep:
    return VadStep(**params)
