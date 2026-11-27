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
from chimera_ml.inference.steps.xai_steps import XAIStep, explain_batch, intervene_batch, modality_available
from chimera_ml.models.utils import get_modality_validity


class MaskedSumModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.grad_enabled: list[bool] = []

    def forward(self, batch: Batch) -> ModelOutput:
        self.calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        total = None

        for name, values in batch.inputs.items():
            mask = batch.get_masks(f"{name}_mask")
            if mask is None:
                valid = torch.ones(values.shape[0], dtype=values.dtype, device=values.device)
            else:
                valid = torch.as_tensor(mask, device=values.device).to(values.dtype)

            current = values * valid[:, None]
            total = current if total is None else total + current

        assert total is not None
        return ModelOutput(preds=total, aux={"inputs": total})


def make_batch(names: tuple[str, ...] = ("audio", "video")) -> Batch:
    values = {
        "audio": torch.tensor([[2.0], [4.0]]),
        "video": torch.tensor([[1.0], [3.0]]),
        "text": torch.tensor([[5.0], [6.0]]),
    }
    return Batch(
        inputs={name: values[name] for name in names},
        targets=None,
        meta={"sample_ids": ["first", "second"]},
    )


def test_remove_intervention_computes_signed_delta_and_preserves_batch() -> None:
    model = MaskedSumModel().eval()
    batch = make_batch()
    original = {name: value.clone() for name, value in batch.inputs.items()}

    artifact = explain_batch(model, batch, modalities=("audio", "video"))

    assert model.calls == 3
    assert model.grad_enabled == [False, False, False]
    assert artifact["samples"][0]["baseline"] == [3.0]
    assert artifact["samples"][0]["modalities"]["audio"]["delta"] == [2.0]
    assert artifact["samples"][0]["modalities"]["video"]["delta"] == [1.0]
    assert torch.equal(batch.inputs["audio"], original["audio"])
    assert batch.masks is None


def test_mask_intervention_creates_availability_mask_and_excludes_values() -> None:
    model = MaskedSumModel().eval()
    batch = make_batch()
    changed = intervene_batch(batch, "audio", "mask")

    assert changed.inputs["audio"].equal(batch.inputs["audio"])
    assert changed.get_masks("audio_mask").tolist() == [False, False]
    assert batch.get_masks("audio_mask") is None

    artifact = explain_batch(model, batch, modalities=("audio",), ablation_policy="mask")
    assert artifact["samples"][0]["modalities"]["audio"]["delta"] == [2.0]


def test_mask_intervention_uses_effective_canonical_masks_over_legacy_masks() -> None:
    model = MaskedSumModel().eval()
    batch = make_batch()
    batch.masks = {
        "audio_mask": torch.tensor([True, True]),
        "video_mask": torch.tensor([True, True]),
    }
    batch.meta = {
        "masks": {
            "audio_mask": torch.tensor([False, False]),
            "video_mask": torch.tensor([False, True]),
        },
        "sample_ids": ["first", "second"],
    }

    changed = intervene_batch(batch, "audio", "mask")
    artifact = explain_batch(model, batch, modalities=("audio",), ablation_policy="mask")

    assert changed.get_masks("audio_mask").tolist() == [False, False]
    assert changed.meta["masks"]["audio_mask"].tolist() == [False, False]
    assert changed.get_masks("video_mask").tolist() == [True, True]
    assert changed.meta["masks"]["video_mask"].tolist() == [False, True]
    assert batch.get_masks("audio_mask").tolist() == [True, True]
    assert batch.meta["masks"]["audio_mask"].tolist() == [False, False]
    assert artifact["samples"][0]["modalities"]["audio"]["delta"] == [2.0]


def test_mask_intervention_preserves_existing_unrelated_masks() -> None:
    batch = make_batch()
    batch.masks = {
        "audio_mask": torch.tensor([True, True]),
        "video_mask": torch.tensor([True, False]),
    }

    changed = intervene_batch(batch, "audio", "mask")

    assert changed.get_masks("audio_mask").tolist() == [False, False]
    assert changed.get_masks("video_mask").tolist() == [True, False]
    assert batch.get_masks("audio_mask").tolist() == [True, True]


def test_modality_availability_matches_fusion_mask_resolution() -> None:
    temporal_batch = Batch(
        inputs={"audio": torch.ones(2, 2, 1)},
        targets=None,
        masks={
            "audio_temporal_mask": torch.tensor([[True, False], [False, True]]),
            "sequence_mask": torch.tensor([[False, False], [True, True]]),
        },
    )
    rank_two_batch = Batch(
        inputs={"audio": torch.ones(2, 1)},
        targets=None,
        masks={"sequence_mask": torch.tensor([True, False])},
    )
    missing_batch = Batch(inputs={"video": torch.ones(2, 1)}, targets=None)

    assert modality_available(temporal_batch, "audio").tolist() == [True, True]
    assert torch.equal(
        modality_available(temporal_batch, "audio"),
        get_modality_validity(temporal_batch, "audio"),
    )
    assert modality_available(rank_two_batch, "audio").tolist() == [True, False]
    assert torch.equal(
        modality_available(rank_two_batch, "audio"),
        get_modality_validity(rank_two_batch, "audio"),
    )
    assert modality_available(missing_batch, "audio").numel() == 0
    assert get_modality_validity(missing_batch, "audio").numel() == 0


def test_zero_intervention_is_rejected() -> None:
    with pytest.raises(ValueError, match="remove, mask"):
        intervene_batch(make_batch(), "audio", "zero")

    with pytest.raises(ValueError, match="remove, mask"):
        explain_batch(MaskedSumModel().eval(), make_batch(), ablation_policy="zero")


def test_missing_samples_are_not_recomputed() -> None:
    model = MaskedSumModel().eval()
    batch = make_batch()
    batch.masks = {"audio_mask": torch.tensor([True, False])}

    artifact = explain_batch(model, batch, modalities=("audio",))

    assert artifact["samples"][0]["modalities"]["audio"]["status"] == "evaluated"
    assert artifact["samples"][1]["modalities"]["audio"]["status"] == "already_missing"
    assert model.calls == 2


def test_score_selector_and_output_mask_are_applied() -> None:
    model = MaskedSumModel().eval()
    artifact = explain_batch(
        model,
        make_batch(),
        modalities=("audio",),
        score_fn=lambda output: output.preds[:, 0],
    )

    assert artifact["samples"][0]["baseline"] == 3.0
    assert artifact["samples"][0]["modalities"]["audio"]["delta"] == 2.0

    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        explain_batch(model, make_batch(), score_fn=lambda output: [1, 2])


def test_model_must_be_in_evaluation_mode() -> None:
    with pytest.raises(ValueError, match="evaluation mode"):
        explain_batch(MaskedSumModel(), make_batch())


@dataclass
class CallbackStep:
    model: MaskedSumModel
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
    model = MaskedSumModel().eval()
    previous_outputs: list[ModelOutput] = []

    def previous(model_arg: nn.Module, batch: Batch, output: ModelOutput) -> None:
        previous_outputs.append(output)

    step = CallbackStep(model, [make_batch()], previous)
    result = XAIStep(step).run(InferenceContext(tmp_path, tmp_path, "cpu", {}))

    assert result.get_artifact("predictions") == {"ok": True}
    assert result.get_artifact("xai")["samples"]
    assert len(previous_outputs) == 1
    assert step.on_prediction is previous


def test_xai_step_collects_union_of_modalities_across_batches(tmp_path: Path) -> None:
    model = MaskedSumModel().eval()
    step = CallbackStep(model, [make_batch(("audio", "video")), make_batch(("video", "text"))])

    result = XAIStep(step).run(InferenceContext(tmp_path, tmp_path, "cpu", {}))
    artifact = result.get_artifact("xai")

    assert artifact["modalities"] == ["audio", "video", "text"]
    assert len(artifact["samples"]) == 4


def test_xai_step_accepts_nested_registered_callback_step(tmp_path: Path) -> None:
    key = "test_xai_callback_step"

    @INFERENCE_STEPS.register(key)
    def create_step(**params: Any) -> CallbackStep:
        return CallbackStep(**params)

    model = MaskedSumModel().eval()
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


def test_model_without_intervention_validator_is_supported() -> None:
    artifact = explain_batch(MaskedSumModel().eval(), make_batch(), modalities=("audio",))
    assert artifact["samples"][0]["modalities"]["audio"]["status"] == "evaluated"
