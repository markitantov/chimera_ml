from dataclasses import dataclass, field
from typing import Any

import torch
from fusion.models.fusion_models import LabelEncoderMeanSimpleFusion
from inference.utils import resolve_checkpoint_path

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference import InferenceContext


@dataclass
class FusionStep:
    batch_size: int = 8
    emotion_class_names: list[str] = field(
        default_factory=lambda: ["neutral", "happy", "sad", "anger", "surprise", "disgust", "fear"]
    )

    sentiment_class_names: list[str] = field(default_factory=lambda: ["negative", "neutral", "positive"])
    _model: Any = field(default=None, init=False, repr=False)

    def run(self, ctx: InferenceContext) -> InferenceContext:
        features = list(ctx.get_artifact("features", []))
        if not features:
            ctx.set_artifact(
                "predictions",
                {
                    "emotion": None,
                    "sentiment": None,
                    "text": "",
                    "segments": [],
                    "status": "no_predictions",
                },
            )

            return ctx

        model = self._load_model(ctx, features[0])
        segment_predictions: list[dict[str, Any]] = []
        emotion_probs_list: list[torch.Tensor] = []
        sentiment_probs_list: list[torch.Tensor] = []
        weights: list[float] = []

        for offset in range(0, len(features), self.batch_size):
            batch_items = features[offset : offset + self.batch_size]
            batch = Batch(
                inputs={
                    "a_features": torch.stack(
                        [torch.as_tensor(item["a_features"], dtype=torch.float32) for item in batch_items]
                    ),
                    "v_features": torch.stack(
                        [torch.as_tensor(item["v_features"], dtype=torch.float32) for item in batch_items]
                    ),
                    "t_features": torch.stack(
                        [torch.as_tensor(item["t_features"], dtype=torch.float32) for item in batch_items]
                    ),
                },
                targets=None,
            )

            batch.inputs = {key: value.to(torch.device(ctx.device)) for key, value in batch.inputs.items()}

            with torch.no_grad():
                logits_batch = model(batch).preds.detach().cpu()

            num_emotions = len(self.emotion_class_names)
            for item, logits in zip(batch_items, logits_batch, strict=True):
                emotion_logits = logits[:num_emotions]
                sentiment_logits = logits[num_emotions : num_emotions + len(self.sentiment_class_names)]
                emotion_probs = torch.softmax(emotion_logits, dim=-1)
                sentiment_probs = torch.softmax(sentiment_logits, dim=-1)
                emotion_index = int(torch.argmax(emotion_probs).item())
                sentiment_index = int(torch.argmax(sentiment_probs).item())
                weight = max(float(item["weight"]), 1e-6)

                emotion_probs_list.append(emotion_probs)
                sentiment_probs_list.append(sentiment_probs)
                weights.append(weight)
                segment_predictions.append(
                    {
                        "index": int(item["index"]),
                        "start_sec": round(float(item["start_sec"]), 6),
                        "end_sec": round(float(item["end_sec"]), 6),
                        "duration_sec": round(float(item["duration_sec"]), 6),
                        "speech_duration_sec": round(float(item["speech_duration_sec"]), 6),
                        "weight": round(weight, 6),
                        "text": str(item["text"]),
                        "has_text": bool(item["has_text"]),
                        "has_speech": bool(item["has_speech"]),
                        "has_faces": bool(item["has_faces"]),
                        "num_windows": int(item["num_windows"]),
                        "emotion": {
                            "label": self.emotion_class_names[emotion_index],
                            "class_index": emotion_index,
                            "probabilities": {
                                name: round(float(emotion_probs[index].item()), 6)
                                for index, name in enumerate(self.emotion_class_names)
                            },
                        },
                        "sentiment": {
                            "label": self.sentiment_class_names[sentiment_index],
                            "class_index": sentiment_index,
                            "probabilities": {
                                name: round(float(sentiment_probs[index].item()), 6)
                                for index, name in enumerate(self.sentiment_class_names)
                            },
                        },
                    }
                )

        weights_tensor = torch.as_tensor(weights, dtype=torch.float32)
        emotion_mean_probs = (torch.stack(emotion_probs_list, dim=0) * weights_tensor[:, None]).sum(
            dim=0
        ) / weights_tensor.sum()
        sentiment_mean_probs = (torch.stack(sentiment_probs_list, dim=0) * weights_tensor[:, None]).sum(
            dim=0
        ) / weights_tensor.sum()
        emotion_index = int(torch.argmax(emotion_mean_probs).item())
        sentiment_index = int(torch.argmax(sentiment_mean_probs).item())

        ctx.set_artifact(
            "predictions",
            {
                "emotion": {
                    "label": self.emotion_class_names[emotion_index],
                    "class_index": emotion_index,
                    "probabilities": {
                        name: round(float(emotion_mean_probs[index].item()), 6)
                        for index, name in enumerate(self.emotion_class_names)
                    },
                },
                "sentiment": {
                    "label": self.sentiment_class_names[sentiment_index],
                    "class_index": sentiment_index,
                    "probabilities": {
                        name: round(float(sentiment_mean_probs[index].item()), 6)
                        for index, name in enumerate(self.sentiment_class_names)
                    },
                },
                "text": " ".join(item["text"] for item in segment_predictions if item["text"]).strip(),
                "segments": segment_predictions,
            },
        )

        return ctx

    def _load_model(self, ctx: InferenceContext, features: dict[str, torch.Tensor]) -> Any:
        if self._model is not None:
            return self._model

        checkpoint_path = resolve_checkpoint_path(ctx, "lefsa")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = LabelEncoderMeanSimpleFusion(
            a_size=tuple(int(x) for x in features["a_features"].shape),
            v_size=tuple(int(x) for x in features["v_features"].shape),
            t_size=tuple(int(x) for x in features["t_features"].shape),
            out_emo=len(self.emotion_class_names),
            out_sen=len(self.sentiment_class_names),
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict)
        model.to(ctx.device)
        model.eval()
        self._model = model

        return model


@INFERENCE_STEPS.register("fusion_step")
def fusion_step(**params: Any) -> FusionStep:
    return FusionStep(**params)
