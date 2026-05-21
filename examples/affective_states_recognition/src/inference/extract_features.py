from dataclasses import dataclass, field
from typing import Any

import torch
from common.utils import find_intersections, slice_audio
from feature_extractors.audio_features import AudioFeatureExtractor
from feature_extractors.text_features import TextFeatureExtractor
from feature_extractors.video_features import VideoFeatureExtractor
from inference.utils import resolve_checkpoint_path
from PIL import Image

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference import InferenceContext


def _extract_face_features(extractor: VideoFeatureExtractor, images: list[Image.Image]) -> torch.Tensor:
    if not images:
        return torch.zeros((0, extractor.output_shape[1]), dtype=torch.float32)

    batch = torch.stack([extractor.preprocessor(image) for image in images], dim=0).to(extractor.device)
    with torch.no_grad():
        features = extractor.model.extract_features(batch)

    return features.detach().cpu()


@dataclass
class ExtractFeaturesStep:
    sample_rate: int = 16000
    win_max_length: int = 3
    win_shift: int = 2
    win_min_length: int = 2
    video_target_fps: int = 10
    text_max_length: int = 48
    audio_model_name: str = "wav2vec2"
    text_model_name: str = "jinaai"
    video_model_name: str = "emoaffectnet"

    _audio_extractor: Any = field(default=None, init=False, repr=False)
    _text_extractor: Any = field(default=None, init=False, repr=False)
    _video_extractor: Any = field(default=None, init=False, repr=False)

    _zero_audio_features: torch.Tensor | None = field(default=None, init=False, repr=False)
    _zero_text_features: torch.Tensor | None = field(default=None, init=False, repr=False)
    _zero_video_features: torch.Tensor | None = field(default=None, init=False, repr=False)

    def run(self, ctx: InferenceContext) -> InferenceContext:
        self.sample_rate = int(ctx.get_artifact("audio_sample_rate"))
        self._init_extractors(ctx)

        audio_waveform = ctx.get_artifact("audio_waveform")
        faces = list(ctx.get_artifact("faces", []))
        segments = list(ctx.get_artifact("segments", []))
        prepared_features: list[dict[str, Any]] = []

        for segment in segments:
            start = int(segment["start"])
            end = int(segment["end"])
            text = str(segment.get("text", "") or "").strip()
            windows = (
                slice_audio(
                    start_time=start,
                    end_time=end,
                    win_max_length=int(self.win_max_length * self.sample_rate),
                    win_shift=int(self.win_shift * self.sample_rate),
                    win_min_length=int(self.win_min_length * self.sample_rate),
                )
                if end > start
                else []
            )

            audio_windows: list[torch.Tensor] = []
            video_windows: list[torch.Tensor] = []
            slot_count = max(int(self.video_target_fps * self.win_max_length), 1)
            slot_duration = 1.0 / float(self.video_target_fps)

            for window in windows:
                window_start = int(window["start"])
                window_end = int(window["end"])
                window_start_sec = float(window_start) / self.sample_rate
                window_end_sec = float(window_end) / self.sample_rate

                speech_segments = find_intersections([window], list(segment.get("speech_segments", [])))
                if speech_segments and window_end > window_start:
                    wave = audio_waveform[window_start:window_end].clone()
                    audio_windows.append(torch.as_tensor(self._audio_extractor(wave), dtype=torch.float32))
                else:
                    audio_windows.append(self._zero_audio_features.clone())

                face_slots: list[Image.Image | None] = [None] * slot_count
                for face in faces:
                    timestamp = float(face["timestamp"])
                    if not (window_start_sec <= timestamp < window_end_sec):
                        continue

                    slot_index = int((timestamp - window_start_sec) // slot_duration)
                    if 0 <= slot_index < slot_count and face_slots[slot_index] is None:
                        face_slots[slot_index] = face["crop_image"]

                video_features = self._zero_video_features.clone()
                present_slots = [(index, image) for index, image in enumerate(face_slots) if image is not None]
                if present_slots:
                    slot_features = _extract_face_features(self._video_extractor, [image for _, image in present_slots])
                    for feature_index, (slot_index, _) in enumerate(present_slots):
                        video_features[slot_index] = slot_features[feature_index]

                video_windows.append(video_features)

            prepared_features.append(
                {
                    "index": int(segment["index"]),
                    "start_sec": float(segment["start_sec"]),
                    "end_sec": float(segment["end_sec"]),
                    "duration_sec": float(segment["duration_sec"]),
                    "speech_duration_sec": float(segment["speech_duration_sec"]),
                    "weight": float(segment["weight"]),
                    "text": text,
                    "has_text": bool(text),
                    "has_speech": bool(segment.get("has_speech", False)),
                    "has_faces": any(
                        float(segment["start_sec"]) <= float(face["timestamp"]) < float(segment["end_sec"])
                        for face in faces
                    ),
                    "num_windows": len(windows),
                    "a_features": torch.stack(audio_windows, dim=0).mean(dim=0)
                    if audio_windows
                    else self._zero_audio_features.clone(),
                    "v_features": torch.stack(video_windows, dim=0).mean(dim=0)
                    if video_windows
                    else self._zero_video_features.clone(),
                    "t_features": torch.as_tensor(self._text_extractor(text), dtype=torch.float32)
                    if text
                    else self._zero_text_features.clone(),
                }
            )

        if not prepared_features:
            prepared_features.append(
                {
                    "index": 0,
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                    "duration_sec": 0.0,
                    "speech_duration_sec": 0.0,
                    "weight": 1.0,
                    "text": "",
                    "has_text": False,
                    "has_speech": False,
                    "has_faces": False,
                    "num_windows": 0,
                    "a_features": self._zero_audio_features.clone(),
                    "v_features": self._zero_video_features.clone(),
                    "t_features": self._zero_text_features.clone(),
                }
            )

        ctx.set_artifact("features", prepared_features)
        return ctx

    def _init_extractors(self, ctx: InferenceContext) -> None:
        if self._audio_extractor is not None and self._text_extractor is not None and self._video_extractor is not None:
            return

        self._audio_extractor = AudioFeatureExtractor(
            sr=self.sample_rate,
            win_max_length=self.win_max_length,
            model_name=self.audio_model_name,
            device=ctx.device,
        )

        self._text_extractor = TextFeatureExtractor(
            max_length=self.text_max_length,
            model_name=self.text_model_name,
            device=ctx.device,
        )

        self._video_extractor = VideoFeatureExtractor(
            win_max_length=self.win_max_length,
            target_fps=self.video_target_fps,
            model_name=self.video_model_name,
            checkpoint_path=resolve_checkpoint_path(ctx, self.video_model_name),
            device=ctx.device,
        )

        self._zero_audio_features = torch.zeros(self._audio_extractor.output_shape, dtype=torch.float32)
        self._zero_text_features = torch.zeros(self._text_extractor.output_shape, dtype=torch.float32)
        self._zero_video_features = torch.zeros(self._video_extractor.output_shape, dtype=torch.float32)


@INFERENCE_STEPS.register("extract_features_step")
def extract_features_step(**params: Any) -> ExtractFeaturesStep:
    return ExtractFeaturesStep(**params)
