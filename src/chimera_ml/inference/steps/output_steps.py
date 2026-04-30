import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference.context import InferenceContext


def serialize_infer_output(payload: dict[str, object]) -> str:
    """Serialize inference output payload to formatted JSON."""
    return json.dumps(payload, indent=2)


@dataclass
class WriteJsonPredictionsStep:
    output_path: str | None = None

    def run(self, ctx: InferenceContext) -> InferenceContext:
        if ctx.predictions is None:
            raise ValueError("No inference output available. Expected 'predictions' artifact.")

        payload = {
            "input": str(ctx.input_path),
            "predictions": ctx.predictions,
        }

        output_path = Path(self.output_path) if self.output_path else ctx.input_path.with_suffix(".json")
        ctx.set_artifact("output_path", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return ctx


@dataclass
class PrintJsonPredictionsStep:
    def run(self, ctx: InferenceContext) -> InferenceContext:
        if ctx.predictions is None:
            raise ValueError("No inference output available. Expected 'predictions' artifact.")

        payload = {
            "input": str(ctx.input_path),
            "predictions": ctx.predictions,
        }

        print(json.dumps(payload, indent=2))
        return ctx


@INFERENCE_STEPS.register("write_json_predictions_step")
def write_json_predictions_step(**params: Any) -> WriteJsonPredictionsStep:
    return WriteJsonPredictionsStep(**params)


@INFERENCE_STEPS.register("print_json_predictions_step")
def print_json_predictions_step(**params: Any) -> PrintJsonPredictionsStep:
    return PrintJsonPredictionsStep(**params)
