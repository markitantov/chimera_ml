from dataclasses import dataclass, field
from typing import Any

import torch
from fusion.models.fusion_models import AVModelV3
from inference.utils import resolve_checkpoint_path

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference import InferenceContext


@dataclass
class FusionStep:
    batch_size: int = 8
    gender_class_names: list[str] = field(default_factory=lambda: ["female", "male"])
    age_scale: float = 100.0
    _model: Any = field(default=None, init=False, repr=False)

    def run(self, ctx: InferenceContext) -> InferenceContext:
        self.age_scale = float(ctx.get_artifact("age_scale"))
        self.gender_class_names = ctx.get_artifact("gender_class_names")

        model = self._load_model(ctx)
        features = ctx.get_artifact("features", [])
        predictions: list[dict[str, Any]] = []

        for offset in range(0, len(features), self.batch_size):
            batch_items = features[offset : offset + self.batch_size]
            batch = Batch(
                inputs={
                    "audio": torch.stack([torch.as_tensor(item["audio"], dtype=torch.float32) for item in batch_items]),
                    "image": torch.stack([torch.as_tensor(item["image"], dtype=torch.float32) for item in batch_items]),
                },
                targets=None,
            )
            batch.inputs = {key: value.to(torch.device(ctx.device)) for key, value in batch.inputs.items()}

            with torch.no_grad():
                outputs = model(batch).preds.detach().cpu()

            gender_logits = outputs[:, : len(self.gender_class_names)]
            gender_probs = torch.softmax(gender_logits, dim=-1)
            age_values = torch.sigmoid(outputs[:, len(self.gender_class_names)]) * self.age_scale

            for item, probs, age in zip(batch_items, gender_probs, age_values, strict=True):
                predicted_index = int(torch.argmax(probs).item())
                predictions.append(
                    {
                        "index": int(item["index"]),
                        "start_sec": round(float(item["start_sec"]), 6),
                        "end_sec": round(float(item["end_sec"]), 6),
                        "gender_probs": {
                            class_name: round(float(probs[class_index].item()), 6)
                            for class_index, class_name in enumerate(self.gender_class_names)
                        },
                        "gender_pred_index": predicted_index,
                        "gender_pred_label": self.gender_class_names[predicted_index],
                        "age_pred": round(float(age.item()), 6),
                        "num_faces": int(item["num_faces"]),
                        "num_frames": int(item["num_frames"]),
                    }
                )

        ctx.set_artifact("window_predictions", predictions)
        return ctx

    def _load_model(self, ctx: InferenceContext) -> Any:
        if self._model is not None:
            return self._model

        checkpoint_path = resolve_checkpoint_path(
            ctx,
            checkpoint_key="fusion",
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = AVModelV3(features_type=ctx.get_artifact("features_type"))
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.to(ctx.device)
        model.eval()
        self._model = model

        return model


@dataclass
class AggregateWindowsStep:
    include_windows: bool = True
    include_debug: bool = True

    def run(self, ctx: InferenceContext) -> InferenceContext:
        window_predictions = ctx.get_artifact("window_predictions", [])
        if not window_predictions:
            ctx.set_artifact(
                "predictions",
                {
                    "gender": None,
                    "age": None,
                    "num_windows": 0,
                    "windows": [],
                    "status": "no_predictions",
                },
            )
            return ctx

        first_window = window_predictions[0]
        gender_class_names = list(first_window["gender_probs"].keys())
        mean_gender_probs = {
            class_name: round(
                float(
                    sum(float(item["gender_probs"][class_name]) for item in window_predictions)
                    / len(window_predictions)
                ),
                6,
            )
            for class_name in gender_class_names
        }

        predicted_label = max(mean_gender_probs, key=mean_gender_probs.get)
        predicted_index = gender_class_names.index(predicted_label)
        mean_age = round(sum(float(item["age_pred"]) for item in window_predictions) / len(window_predictions), 6)

        payload: dict[str, Any] = {
            "gender": {
                "label": predicted_label,
                "class_index": predicted_index,
                "probabilities": mean_gender_probs,
            },
            "age": {"value": mean_age},
            "num_windows": len(window_predictions),
        }

        if self.include_windows:
            payload["windows"] = window_predictions

        if self.include_debug:
            sample_rate = int(ctx.get_artifact("audio_sample_rate", 16000))
            payload["debug"] = {
                "speech_segments_sec": [
                    {
                        "start_sec": round(float(segment["start"]) / sample_rate, 6),
                        "end_sec": round(float(segment["end"]) / sample_rate, 6),
                    }
                    for segment in ctx.get_artifact("vad_segments", [])
                ],
                "selected_face_track": ctx.get_artifact("selected_face_track"),
            }

        ctx.set_artifact("predictions", payload)
        return ctx


@INFERENCE_STEPS.register("fusion_step")
def fusion_step(**params: Any) -> FusionStep:
    return FusionStep(**params)


@INFERENCE_STEPS.register("aggregate_windows_step")
def aggregate_windows_step(**params: Any) -> AggregateWindowsStep:
    return AggregateWindowsStep(**params)
