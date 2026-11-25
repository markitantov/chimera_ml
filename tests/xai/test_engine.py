import json
from dataclasses import dataclass

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.inference import InferenceContext, build_inference_step
from chimera_ml.inference.steps.xai_steps import explain_batch


class MaskAwareModel(torch.nn.Module):
    def forward(self, batch: Batch) -> ModelOutput:
        values = []
        for name in ("audio", "video"):
            mask = batch.get_masks(f"{name}_mask")
            values.append(batch.inputs[name].flatten(1).sum(1) * torch.as_tensor(mask).to(batch.inputs[name]))
        return ModelOutput(torch.stack(values, dim=-1).sum(dim=-1, keepdim=True))


def make_batch() -> Batch:
    return Batch(
        inputs={"audio": torch.tensor([[2.0]]), "video": torch.tensor([[3.0]])},
        targets=None,
        masks={"audio_mask": torch.ones(1, dtype=torch.bool), "video_mask": torch.ones(1, dtype=torch.bool)},
        meta={"sample_ids": ["sample-1"]},
    )


def test_explain_batch_uses_raw_zero_intervention_without_mutating_batch():
    model = MaskAwareModel()
    batch = make_batch()
    original = {key: value.clone() for key, value in batch.inputs.items()}
    result = explain_batch(model, batch, score_fn=lambda preds: preds, modalities=("audio", "video"))

    assert result["samples"][0]["modalities"]["audio"]["delta"] == [2.0]
    assert result["samples"][0]["modalities"]["video"]["delta"] == [3.0]
    assert torch.equal(batch.inputs["audio"], original["audio"])
    json.dumps(result, allow_nan=False)


@dataclass
class PredictionStep:
    on_prediction: object = None

    def run(self, ctx: InferenceContext) -> InferenceContext:
        batch = make_batch()
        output = MaskAwareModel()(batch)
        if self.on_prediction is not None:
            self.on_prediction(MaskAwareModel(), batch, output)
        ctx.set_artifact("predictions", {"ok": True})
        return ctx


@INFERENCE_STEPS.register("test_prediction_step")
def test_prediction_step(**params: object) -> PredictionStep:
    return PredictionStep(**params)


def test_xai_step_wraps_nested_registered_step_and_preserves_predictions(tmp_path):
    step = build_inference_step({"name": "xai_step", "params": {"step": {"name": "test_prediction_step"}}})
    ctx = InferenceContext(tmp_path / "input", tmp_path, "cpu", {})
    result = step.run(ctx)

    assert result.get_artifact("predictions") == {"ok": True}
    assert len(result.get_artifact("xai")["samples"]) == 1


def test_xai_step_is_registered():
    assert "xai_step" in INFERENCE_STEPS._items


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, batch: Batch) -> ModelOutput:
        self.seen.append(batch)
        return ModelOutput(sum(batch.inputs.values()))


def test_zero_mask_and_remove_interventions_are_explicit():
    batch = Batch({"audio": torch.ones(1, 1), "video": torch.full((1, 1), 2.0)}, None)
    for policy in ("zero", "mask", "remove"):
        model = RecordingModel()
        result = explain_batch(model, batch, modalities=("audio",), ablation_policy=policy)
        assert result["samples"][0]["modalities"]["audio"]["status"] == "evaluated"
        assert batch.inputs["audio"].item() == 1.0


def test_callback_is_restored_after_wrapped_step_failure(tmp_path):
    from chimera_ml.inference.steps.xai_steps import XAIStep

    def previous(model, batch, output):
        return None

    class FailingStep:
        on_prediction = previous

        def run(self, ctx):
            raise RuntimeError("failure")

    step = XAIStep(FailingStep())
    ctx = InferenceContext(tmp_path / "input", tmp_path, "cpu", {})
    try:
        step.run(ctx)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected wrapped step failure")
    assert step.step.on_prediction.__func__ is previous
