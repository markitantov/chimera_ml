from dataclasses import dataclass, field
from typing import Any

import torch
from common.utils import FeaturesType, find_intersections, slice_audio
from fusion.data.feature_extractors import AudioFeatureExtractor, ImageFeatureExtractor
from inference.utils import resolve_checkpoint_path
from PIL import Image

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference import InferenceContext


@dataclass
class BuildWindowsStep:
    sample_rate: int = 16000
    win_max_length: float = 4.0
    win_shift: float = 2.0
    win_min_length: float = 1.0
    fps: float = 1.0

    def run(self, ctx: InferenceContext) -> InferenceContext:
        audio_num_samples = int(ctx.get_artifact("audio_num_samples", 0))
        self.sample_rate = int(ctx.get_artifact("audio_sample_rate"))
        self.fps = float(ctx.get_artifact("fps"))
        ctx.set_artifact("win_max_length", self.win_max_length)
        vad_segments = list(ctx.get_artifact("vad_segments", []))
        faces = list(ctx.get_artifact("faces", []))

        windows = slice_audio(
            start_time=0,
            end_time=audio_num_samples,
            win_max_length=int(self.win_max_length * self.sample_rate),
            win_shift=int(self.win_shift * self.sample_rate),
            win_min_length=int(self.win_min_length * self.sample_rate),
        )

        prepared_windows: list[dict[str, Any]] = []
        slot_count = max(round(self.win_max_length * self.fps), 1)
        slot_duration_sec = 1.0 / self.fps

        for index, window in enumerate(windows):
            start_sample = int(window["start"])
            end_sample = int(window["end"])
            start_sec = float(start_sample) / self.sample_rate
            end_sec = float(end_sample) / self.sample_rate

            speech_segments = find_intersections([window], vad_segments)
            window_faces = [item for item in faces if start_sec <= float(item["timestamp"]) < end_sec]
            face_slots: list[Image.Image | None] = [None] * slot_count

            for face in window_faces:
                rel_sec = float(face["timestamp"]) - start_sec
                slot_index = int(rel_sec // slot_duration_sec)
                if 0 <= slot_index < slot_count and face_slots[slot_index] is None:
                    face_slots[slot_index] = face["crop_image"]

            prepared_windows.append(
                {
                    "index": index,
                    "start": start_sample,
                    "end": end_sample,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "has_speech": bool(speech_segments),
                    "speech_segments": speech_segments,
                    "face_slots": face_slots,
                }
            )

        ctx.set_artifact("windows", prepared_windows)
        return ctx


@dataclass
class ExtractFeaturesStep:
    sample_rate: int = 16000
    win_max_length: float = 4.0
    features_type: FeaturesType = FeaturesType.INTERMEDIATE
    gender_class_names: list[str] = field(default_factory=lambda: ["female", "male"])
    age_scale: float = 100.0

    _audio_extractor: Any = field(default=None, init=False, repr=False)
    _image_extractor: Any = field(default=None, init=False, repr=False)
    _zero_audio_features: torch.Tensor | None = field(default=None, init=False, repr=False)
    _zero_image_features: torch.Tensor | None = field(default=None, init=False, repr=False)

    def run(self, ctx: InferenceContext) -> InferenceContext:
        self.sample_rate = int(ctx.get_artifact("audio_sample_rate"))
        self.win_max_length = int(ctx.get_artifact("win_max_length"))
        self.device = ctx.device
        ctx.set_artifact("features_type", self.features_type)
        ctx.set_artifact("gender_class_names", self.gender_class_names)
        ctx.set_artifact("age_scale", self.age_scale)

        self._init_extractors(ctx)

        audio_waveform = ctx.get_artifact("audio_waveform")
        if audio_waveform is None:
            raise ValueError("Expected 'audio_waveform' artifact from extract_audio_step.")

        features: list[dict[str, Any]] = []
        for window in ctx.get_artifact("windows", []):
            wave = audio_waveform[int(window["start"]) : int(window["end"])].clone()
            audio_features = (
                self._audio_extractor(wave) if bool(window["has_speech"]) else self._zero_audio_features.clone()
            )

            image_features = self._zero_image_features.clone()
            present_slots = [
                (slot_index, image) for slot_index, image in enumerate(window["face_slots"]) if image is not None
            ]

            if present_slots:
                slot_indices = [slot_index for slot_index, _ in present_slots]
                slot_images = [image for _, image in present_slots]
                slot_features = self._image_extractor(slot_images)[: len(slot_indices)]
                image_features[slot_indices] = slot_features

            features.append(
                {
                    "index": int(window["index"]),
                    "start_sec": float(window["start_sec"]),
                    "end_sec": float(window["end_sec"]),
                    "audio": audio_features,
                    "image": image_features,
                    "num_faces": sum(image is not None for image in window["face_slots"]),
                    "num_frames": len(window["face_slots"]),
                    "has_speech": bool(window["has_speech"]),
                }
            )

        ctx.set_artifact("features", features)
        return ctx

    def _init_extractors(self, ctx: InferenceContext) -> None:
        if self._audio_extractor is not None and self._image_extractor is not None:
            return

        audio_checkpoint_path = resolve_checkpoint_path(ctx, checkpoint_key="audio")

        self._audio_extractor = AudioFeatureExtractor(
            hf_model_name="facebook/wav2vec2-large-robust",
            checkpoint_path=audio_checkpoint_path,
            features_type=self.features_type,
            sr=self.sample_rate,
            win_max_length=self.win_max_length,
            gender_num_classes=len(self.gender_class_names),
            device=self.device,
        )

        image_checkpoint_path = resolve_checkpoint_path(
            ctx,
            checkpoint_key="image",
        )

        self._image_extractor = ImageFeatureExtractor(
            hf_model_name="nateraw/vit-age-classifier",
            checkpoint_path=image_checkpoint_path,
            features_type=self.features_type,
            win_max_length=self.win_max_length,
            device=self.device,
        )

        dummy_wave = torch.zeros(self.sample_rate * self.win_max_length, dtype=torch.float32)
        self._zero_audio_features = torch.zeros_like(self._audio_extractor(dummy_wave))
        dummy_image = Image.new("RGB", (224, 224), color=0)
        self._zero_image_features = torch.zeros_like(self._image_extractor([dummy_image]))


@INFERENCE_STEPS.register("build_windows_step")
def build_windows_step(**params: Any) -> BuildWindowsStep:
    return BuildWindowsStep(**params)


@INFERENCE_STEPS.register("extract_features_step")
def extract_features_step(**params: Any) -> ExtractFeaturesStep:
    return ExtractFeaturesStep(**params)
