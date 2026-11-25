import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.inference import InferenceContext, InferencePipeline, build_inference_step
from chimera_ml.inference.pipeline import InferenceGraphNode
from chimera_ml.inference.steps.xai_steps import XAIStep, _availability, _intervene, explain_batch


class CountingModel(nn.Module):
    def __init__(self, *, fail=False):
        super().__init__()
        self.calls = 0
        self.seen_models = []
        self.fail = fail

    def forward(self, batch: Batch) -> ModelOutput:
        self.calls += 1
        self.seen_models.append(self)
        if self.fail:
            raise RuntimeError("model failure")
        audio = batch.inputs.get("audio", torch.zeros_like(batch.inputs["video"]))
        video = batch.inputs["video"]
        preds = audio.squeeze(-1) + video.squeeze(-1)
        diagnostics = torch.arange(preds.shape[0], dtype=torch.float32, device=preds.device).unsqueeze(-1)
        return ModelOutput(preds, {"diagnostic": diagnostics})


def make_batch(*, sample_ids=None, mixed=True, temporal=False) -> Batch:
    audio = torch.tensor([[[2.0], [4.0]], [[0.0], [0.0]]])
    video = torch.tensor([[[3.0], [5.0]], [[7.0], [11.0]]])
    masks = {
        "audio_mask": torch.tensor([True, mixed]),
        "video_mask": torch.tensor([True, True]),
        "audio_temporal_mask": torch.tensor([[True, True], [False, False]]),
        "video_temporal_mask": torch.tensor([[True, True], [True, True]]),
    }
    if temporal:
        masks["sequence_mask"] = torch.tensor([[True, False], [True, True]])
    return Batch(
        {"audio": audio, "video": video},
        None,
        masks,
        {"sample_ids": sample_ids} if sample_ids is not None else {},
    )


def eval_model(model: nn.Module) -> nn.Module:
    model.eval()
    return model


def test_explain_batch_reuses_baseline_and_preserves_inputs():
    model = eval_model(CountingModel())
    batch = make_batch(mixed=False)
    original = {key: value.clone() for key, value in batch.inputs.items()}
    result = explain_batch(model, batch, modalities=("audio", "video"))

    assert model.calls == 3
    assert result["samples"][0]["modalities"]["audio"]["delta"] == [2.0, 4.0]
    assert torch.equal(batch.inputs["audio"], original["audio"])
    json.dumps(result, allow_nan=False)


def test_score_fn_receives_model_output_and_is_validated():
    model = eval_model(CountingModel())
    batch = make_batch(mixed=False)
    seen = []

    def score(output: ModelOutput) -> torch.Tensor:
        seen.append(output)
        return output.preds[:, :1]

    explain_batch(model, batch, score_fn=score, modalities=("audio",))
    assert seen
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        explain_batch(model, batch, score_fn=lambda output: [1, 2], modalities=("audio",))
    with pytest.raises(ValueError, match="batch-aligned"):
        explain_batch(model, batch, score_fn=lambda output: torch.ones(1), modalities=("audio",))


def test_tensor_diagnostics_are_json_safe_and_batch_aligned():
    model = eval_model(CountingModel())
    result = explain_batch(model, make_batch(mixed=False), diagnostic_keys=("diagnostic",))
    assert result["samples"][1]["diagnostics"]["diagnostic"] == [1.0]
    json.dumps(result, allow_nan=False)

    class BadDiagnostic(CountingModel):
        def forward(self, batch):
            output = super().forward(batch)
            output.aux["diagnostic"] = torch.ones(1, 2)
            return output

    with pytest.raises(ValueError, match="batch-aligned"):
        explain_batch(eval_model(BadDiagnostic()), make_batch(mixed=False), diagnostic_keys=("diagnostic",))


def test_mixed_and_entirely_missing_modalities_use_already_missing_without_forward():
    model = eval_model(CountingModel())
    batch = make_batch(mixed=True)
    result = explain_batch(model, batch, modalities=("audio", "missing"))

    assert model.calls == 2
    assert result["samples"][0]["modalities"]["audio"]["status"] == "evaluated"
    assert result["samples"][1]["modalities"]["audio"]["status"] == "already_missing"
    assert all(sample["modalities"]["missing"]["status"] == "already_missing" for sample in result["samples"])


def test_temporal_validity_uses_null_for_non_common_positions():
    model = eval_model(CountingModel())
    result = explain_batch(
        model, make_batch(mixed=False, temporal=True), modalities=("audio",), output_mask_key="sequence_mask"
    )

    assert result["samples"][0]["baseline"] == [5.0, None]
    assert result["samples"][0]["modalities"]["audio"]["delta"] == [2.0, None]


def test_remove_intervention_removes_only_modality_masks():
    batch = Batch(
        {"audio": torch.ones(1, 1), "video": torch.ones(1, 1)},
        None,
        {"audio_mask": torch.ones(1), "audio_temporal_mask": torch.ones(1, 1), "video_mask": torch.ones(1)},
        {"masks": {"audio": torch.ones(1), "video": torch.ones(1)}},
    )
    changed = _intervene(batch, "audio", "remove")
    assert "audio" not in changed.inputs
    assert changed.get_masks("audio_mask") is None
    assert changed.get_masks("audio_temporal_mask") is None
    assert changed.get_masks("video_mask") is not None
    assert batch.get_masks("audio_mask") is not None


def test_model_must_already_be_in_eval_mode_without_flag_mutation():
    model = CountingModel()
    with pytest.raises(ValueError, match="evaluation mode"):
        explain_batch(model, make_batch())
    assert model.training

    model.eval()
    child = nn.BatchNorm1d(1)
    child.train()
    model.add_module("child", child)
    with pytest.raises(ValueError, match="nested"):
        explain_batch(model, make_batch())
    assert child.training


@dataclass
class CallbackStep:
    model: CountingModel
    batches: list[Batch]
    on_prediction: object = None
    return_new_context: bool = False

    def run(self, ctx: InferenceContext) -> InferenceContext:
        target = (
            InferenceContext(ctx.input_path, ctx.work_dir, ctx.device, ctx.config, dict(ctx.artifacts))
            if self.return_new_context
            else ctx
        )
        for batch in self.batches:
            output = self.model(batch)
            if self.on_prediction is not None:
                self.on_prediction(self.model, batch, output)
        target.set_artifact("predictions", {"ok": True})
        return target


def test_xai_wrapper_uses_returned_context_and_restores_callback(tmp_path: Path):
    model = eval_model(CountingModel())
    previous_calls = []

    def previous(model_arg, batch, output):
        previous_calls.append((model_arg, output))

    step = CallbackStep(model, [make_batch(mixed=False)], previous, return_new_context=True)
    wrapped = XAIStep(step)
    ctx = InferenceContext(tmp_path / "input", tmp_path, "cpu", {})
    result = wrapped.run(ctx)

    assert result is not ctx
    assert result.get_artifact("predictions") == {"ok": True}
    assert "xai" in result.artifacts
    assert previous_calls[0][0] is model
    assert step.on_prediction is previous


def test_multiple_callbacks_fallback_ids_and_repeated_runs(tmp_path: Path):
    model = eval_model(CountingModel())
    step = CallbackStep(model, [make_batch(mixed=False), make_batch(mixed=False)])
    wrapped = XAIStep(step)
    ctx = InferenceContext(tmp_path / "input", tmp_path, "cpu", {})

    first = wrapped.run(ctx).get_artifact("xai")
    second = wrapped.run(ctx).get_artifact("xai")
    assert [sample["sample_id"] for sample in first["samples"]] == ["batch-0-0", "batch-0-1", "batch-1-0", "batch-1-1"]
    assert len(second["samples"]) == 4
    assert first["batches"] == second["batches"] == 2


def test_callback_contract_and_failure_restoration(tmp_path: Path):
    class InvalidStep:
        on_prediction = 3

        def run(self, ctx):
            return ctx

    with pytest.raises(TypeError, match="callable or None"):
        XAIStep(InvalidStep()).run(InferenceContext(tmp_path, tmp_path, "cpu", {}))

    model = eval_model(CountingModel(fail=True))
    step = CallbackStep(model, [make_batch(mixed=False)])

    def original(model, batch, output):
        return None

    step.on_prediction = original
    with pytest.raises(RuntimeError, match="model failure"):
        XAIStep(step).run(InferenceContext(tmp_path, tmp_path, "cpu", {}))
    assert step.on_prediction is original


def test_nonfinite_values_fail_strict_json():
    class NonFinite(CountingModel):
        def forward(self, batch):
            output = super().forward(batch)
            output.preds[0, 0] = float("nan")
            return output

    with pytest.raises(ValueError, match="Out of range"):
        explain_batch(eval_model(NonFinite()), make_batch(mixed=False), modalities=("audio",))


@INFERENCE_STEPS.register("test_universal_callback_step")
def universal_callback_step(**params):
    return CallbackStep(**params)


def test_registered_wrapper_preserves_predictions_and_sequential_dag(tmp_path: Path):
    model = eval_model(CountingModel())
    step = build_inference_step(
        {
            "name": "xai_step",
            "params": {
                "step": {
                    "name": "test_universal_callback_step",
                    "params": {"model": model, "batches": [make_batch(mixed=False)]},
                }
            },
        }
    )
    ctx = InferenceContext(tmp_path, tmp_path, "cpu", {})
    result = step.run(ctx)
    assert result.get_artifact("predictions") == {"ok": True}
    assert result.get_artifact("xai")["samples"]

    first = CallbackStep(model, [make_batch(mixed=False)])
    second = CallbackStep(model, [make_batch(mixed=False)])
    pipeline = InferencePipeline(
        [
            InferenceGraphNode("first", first),
            InferenceGraphNode("second", second, ("first",)),
        ]
    )
    result = pipeline.run(InferenceContext(tmp_path, tmp_path, "cpu", {}))
    assert result.get_artifact("predictions") == {"ok": True}


def test_availability_uses_dedicated_temporal_precedence_and_shared_sequence_mask():
    batch = Batch(
        {"audio": torch.ones(2, 2, 1)},
        None,
        {
            "audio_mask": torch.ones(2, 2, dtype=torch.bool),
            "audio_temporal_mask": torch.zeros(2, 2, dtype=torch.bool),
            "sequence_mask": torch.ones(2, 2, dtype=torch.bool),
        },
    )
    assert not _availability(batch, "audio", 2).any()

    batch.masks.pop("audio_temporal_mask")
    batch.masks["sequence_mask"] = torch.tensor([[True, False], [False, False]])
    assert torch.equal(_availability(batch, "audio", 2), torch.tensor([True, False]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_availability_handles_cpu_and_cuda_masks():
    cpu_input = Batch(
        {"audio": torch.ones(2, 1)},
        None,
        {"audio_mask": torch.tensor([True, False], device="cuda")},
    )
    assert torch.equal(_availability(cpu_input, "audio", 2), torch.tensor([True, False]))

    cuda_input = Batch(
        {"audio": torch.ones(2, 1, device="cuda")},
        None,
        {"audio_mask": torch.tensor([True, False])},
    )
    result = _availability(cuda_input, "audio", 2)
    assert result.device.type == "cuda"
    assert torch.equal(result.cpu(), torch.tensor([True, False]))


def test_validity_rejection_and_no_common_positions_are_unsupported():
    model = eval_model(CountingModel())
    batch = make_batch(mixed=False)
    rejected = explain_batch(model, batch, modalities=("audio",), validity_fn=lambda name, b, output: False)
    assert rejected["samples"][0]["modalities"]["audio"]["status"] == "unsupported"
    assert "validity_fn" in rejected["samples"][0]["modalities"]["audio"]["reason"]

    batch.masks["sequence_mask"] = torch.zeros(2, 2, dtype=torch.bool)
    no_common = explain_batch(model, batch, modalities=("audio",), output_mask_key="sequence_mask")
    assert no_common["samples"][0]["modalities"]["audio"]["status"] == "unsupported"
    assert "common valid" in no_common["samples"][0]["modalities"]["audio"]["reason"]


def test_remove_intervention_clears_nested_legacy_metadata_masks():
    batch = Batch(
        {"audio": torch.ones(1, 1), "video": torch.ones(1, 1)},
        None,
        None,
        {"masks": {"mask": {"audio": torch.ones(1), "video": torch.ones(1)}}},
    )
    changed = _intervene(batch, "audio", "remove")
    assert changed.get_masks("audio_mask") is None
    assert changed.get_masks("video_mask") is not None
    assert changed.meta["masks"]["mask"].get("audio") is None


def test_predictions_are_unchanged_with_xai(tmp_path: Path):
    batch = make_batch(mixed=False)
    expected = eval_model(CountingModel())(batch).preds.tolist()
    seen = []

    def callback(model, batch_arg, output):
        seen.append(output.preds.detach().clone())

    model = eval_model(CountingModel())
    XAIStep(CallbackStep(model, [batch], callback)).run(InferenceContext(tmp_path, tmp_path, "cpu", {}))
    assert seen[0].tolist() == expected


@dataclass
class ArtifactStep:
    key: str
    on_prediction: object = None

    def run(self, ctx: InferenceContext) -> InferenceContext:
        ctx.set_artifact(self.key, True)
        return ctx


def test_xai_step_runs_inside_sequential_and_parallel_dags(tmp_path: Path):
    model = eval_model(CountingModel())
    sequential = InferencePipeline(
        [
            InferenceGraphNode("xai", XAIStep(CallbackStep(model, [make_batch(mixed=False)]))),
            InferenceGraphNode("after", ArtifactStep("after"), ("xai",)),
        ]
    )
    result = sequential.run(InferenceContext(tmp_path, tmp_path, "cpu", {}))
    assert result.get_artifact("xai")["samples"]
    assert result.get_artifact("after") is True

    parallel = InferencePipeline(
        [
            InferenceGraphNode("xai", XAIStep(CallbackStep(model, [make_batch(mixed=False)]))),
            InferenceGraphNode("other", ArtifactStep("other")),
        ],
        parallel_mode=True,
    )
    result = parallel.run(InferenceContext(tmp_path, tmp_path, "cpu", {}))
    assert result.get_artifact("xai")["samples"]
    assert result.get_artifact("other") is True
