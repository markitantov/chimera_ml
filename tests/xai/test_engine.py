from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.inference import InferenceContext, build_inference_step
from chimera_ml.inference.steps.xai_steps import XAIStep, explain_batch, summarize_explanations


class SumModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, batch: Batch) -> ModelOutput:
        self.calls += 1
        audio = batch.inputs.get("audio")
        video = batch.inputs.get("video")
        reference = audio if audio is not None else video
        assert reference is not None
        audio = torch.zeros_like(reference) if audio is None else audio
        video = torch.zeros_like(reference) if video is None else video
        return ModelOutput(preds=audio + video, aux={"attention": video, "beta_audio": audio})


def make_batch() -> Batch:
    return Batch(
        inputs={"audio": torch.tensor([[2.0], [4.0]]), "video": torch.tensor([[1.0], [3.0]])},
        targets=None,
        meta={"sample_ids": ["first", "second"]},
    )


def test_explain_batch_computes_leave_one_modality_delta() -> None:
    model = SumModel().eval()
    batch = make_batch()
    artifact = explain_batch(model, batch, modalities=("audio", "video"))

    assert model.calls == 3
    assert artifact["samples"][0]["baseline"] == [3.0]
    assert artifact["samples"][0]["modalities"]["audio"]["delta"] == [2.0]
    assert artifact["samples"][0]["modalities"]["video"]["delta"] == [1.0]
    assert artifact["samples"][0]["aux"]["beta_audio"] == [2.0]
    assert torch.equal(batch.inputs["audio"], torch.tensor([[2.0], [4.0]]))


def test_score_selector_and_output_mask_are_applied() -> None:
    model = SumModel().eval()
    batch = make_batch()
    artifact = explain_batch(model, batch, modalities=("audio",), score_fn=lambda output: output.preds[:, 0])

    assert artifact["samples"][0]["baseline"] == 3.0
    assert artifact["samples"][0]["modalities"]["audio"]["delta"] == 2.0

    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        explain_batch(model, batch, score_fn=lambda output: [1, 2])


def test_missing_samples_are_not_recomputed() -> None:
    model = SumModel().eval()
    batch = make_batch()
    batch.masks = {"audio_mask": torch.tensor([True, False])}
    artifact = explain_batch(model, batch, modalities=("audio",))

    assert artifact["samples"][0]["modalities"]["audio"]["status"] == "evaluated"
    assert artifact["samples"][1]["modalities"]["audio"]["status"] == "already_missing"
    assert model.calls == 2


def test_anchor_ablation_is_marked_unsupported() -> None:
    model = SumModel().eval()
    model.anchor_modality = "audio"
    artifact = explain_batch(model, make_batch(), modalities=("audio",))

    record = artifact["samples"][0]["modalities"]["audio"]
    assert record["status"] == "unsupported"
    assert record["reason"] == "anchor modality cannot be removed"


def test_model_must_be_in_evaluation_mode() -> None:
    model = SumModel()
    with pytest.raises(ValueError, match="evaluation mode"):
        explain_batch(model, make_batch())


@dataclass
class CallbackStep:
    model: SumModel
    batches: list[Batch]
    on_prediction: Any = None

    def run(self, ctx: InferenceContext) -> InferenceContext:
        for batch in self.batches:
            output = self.model(batch)
            if self.on_prediction is not None:
                self.on_prediction(self.model, batch, output)

        ctx.set_artifact("predictions", {"ok": True})
        return ctx


def test_xai_step_collects_callback_and_restores_previous_callback(tmp_path: Path) -> None:
    model = SumModel().eval()
    previous_calls: list[ModelOutput] = []

    def previous(model_arg: nn.Module, batch: Batch, output: ModelOutput) -> None:
        previous_calls.append(output)

    step = CallbackStep(model, [make_batch()], previous)
    result = XAIStep(step).run(InferenceContext(tmp_path, tmp_path, "cpu", {}))

    assert result.get_artifact("predictions") == {"ok": True}
    assert result.get_artifact("xai")["samples"]
    assert len(previous_calls) == 1
    assert step.on_prediction is previous
    assert model.calls == 3


def test_xai_step_accepts_nested_registered_callback_step(tmp_path: Path) -> None:
    key = "test_xai_callback_step"

    @INFERENCE_STEPS.register(key)
    def create_step(**params: Any) -> CallbackStep:
        return CallbackStep(**params)

    model = SumModel().eval()
    wrapped = build_inference_step(
        {
            "name": "xai_step",
            "params": {
                "step": {"name": key, "params": {"model": model, "batches": [make_batch()]}},
            },
        }
    )

    result = wrapped.run(InferenceContext(tmp_path, tmp_path, "cpu", {}))
    assert result.get_artifact("xai")["samples"]


def test_summary_ranks_modalities_by_absolute_delta() -> None:
    artifact = explain_batch(SumModel().eval(), make_batch(), modalities=("audio", "video"))
    summary = summarize_explanations(artifact)

    assert summary["modalities"]["audio"]["mean_absolute_delta"] == 3.0
    assert summary["modalities"]["video"]["mean_absolute_delta"] == 2.0
    assert summary["ranking"] == ["audio", "video"]


def test_intervention_updates_nested_legacy_masks() -> None:
    batch = Batch(
        inputs={"audio": torch.ones(1, 1), "video": torch.ones(1, 1)},
        targets=None,
        meta={"masks": {"mask": {"audio": torch.ones(1, dtype=torch.bool)}}},
    )

    from chimera_ml.inference.steps.xai_steps import intervene_batch

    changed = intervene_batch(batch, "audio", "mask")

    assert changed.get_masks("audio_mask").tolist() == [False]
    assert batch.get_masks("audio_mask").tolist() == [True]
