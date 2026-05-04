from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference import InferenceContext
from chimera_ml.training.builders import build_model

from inference.utils import load_model

GENDER_CLASS_NAMES = ["female", "male"]
AGE_SCALE = 100.0


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


@dataclass
class PredictStep:
    checkpoint: str | dict[str, Any]
    batch_size: int = 8
    _model: Any = field(default=None, init=False, repr=False)

    def run(self, ctx: InferenceContext) -> InferenceContext:
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

            gender_logits = outputs[:, : len(GENDER_CLASS_NAMES)]
            gender_probs = torch.softmax(gender_logits, dim=-1)
            age_values = torch.sigmoid(outputs[:, len(GENDER_CLASS_NAMES)]) * AGE_SCALE

            for item, probs, age in zip(batch_items, gender_probs, age_values, strict=True):
                predicted_index = int(torch.argmax(probs).item())
                predictions.append(
                    {
                        "index": int(item["index"]),
                        "start_sec": _round(item["start_sec"]),
                        "end_sec": _round(item["end_sec"]),
                        "gender_probs": {
                            class_name: _round(probs[class_index].item())
                            for class_index, class_name in enumerate(GENDER_CLASS_NAMES)
                        },
                        "gender_pred_index": predicted_index,
                        "gender_pred_label": GENDER_CLASS_NAMES[predicted_index],
                        "age_pred": _round(age.item()),
                        "num_faces": int(item["num_faces"]),
                        "num_frames": int(item["num_frames"]),
                    }
                )

        ctx.set_artifact("window_predictions", predictions)
        return ctx

    def _load_model(self, ctx: InferenceContext) -> Any:
        if self._model is not None:
            return self._model

        model = build_model(
            {
                "name": "agender_multimodal_model_v3",
                "params": {
                    "features_type": "INTERMEDIATE",
                    "include_mask": False,
                },
            }
        )

        checkpoint_ref = self.checkpoint
        if isinstance(checkpoint_ref, dict):
            checkpoint_ref = str(checkpoint_ref["local_path"])

        checkpoint_path = Path(str(checkpoint_ref))
        if not checkpoint_path.exists():
            checkpoint_path = Path(str(load_model(str(checkpoint_ref), cache_dir=ctx.work_dir / "model_cache")))

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict, strict=True)
        model.to(torch.device(ctx.device))
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
            class_name: _round(
                sum(float(item["gender_probs"][class_name]) for item in window_predictions) / len(window_predictions)
            )
            for class_name in gender_class_names
        }
        predicted_label = max(mean_gender_probs, key=mean_gender_probs.get)
        predicted_index = gender_class_names.index(predicted_label)
        mean_age = _round(sum(float(item["age_pred"]) for item in window_predictions) / len(window_predictions))

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
                        "start_sec": _round(float(segment["start"]) / sample_rate),
                        "end_sec": _round(float(segment["end"]) / sample_rate),
                    }
                    for segment in ctx.get_artifact("vad_segments", [])
                ],
                "selected_face_track": ctx.get_artifact("selected_face_track"),
            }

        ctx.set_artifact("predictions", payload)
        return ctx


@INFERENCE_STEPS.register("predict")
def predict(**params: Any) -> PredictStep:
    return PredictStep(**params)


@INFERENCE_STEPS.register("aggregate_windows")
def aggregate_windows(**params: Any) -> AggregateWindowsStep:
    return AggregateWindowsStep(**params)
