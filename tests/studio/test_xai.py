from pathlib import Path
from uuid import uuid4

import torch
from torch import nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.studio.inference import run_inference
from chimera_ml.studio.timeline import build_xai_timeline_html
from chimera_ml.studio.xai import extract_xai_timeline, render_xai_overview


def test_xai_overview_preserves_statuses_nulls_and_signed_deltas() -> None:
    xai = {
        "ablation_policy": "remove",
        "samples": [
            {
                "sample_id": "s-1",
                "baseline": [0.6, 2.0],
                "modalities": {
                    "audio": {"status": "evaluated", "delta": [-0.2, None]},
                    "video": {"status": "already_missing", "delta": None},
                    "text": {"status": "unsupported", "reason": "cannot remove", "delta": None},
                },
            }
        ],
    }
    markup = render_xai_overview(xai)
    assert "Ablation policy" in markup
    assert "-0.2" in markup
    assert "null" in markup
    assert "already_missing" in markup
    assert "unsupported" in markup
    assert "cannot remove" in markup
    assert "causal" not in markup.lower()


def test_xai_timeline_requires_explicit_sample_id_mapping() -> None:
    xai = {
        "samples": [
            {"sample_id": "second", "modalities": {"audio": {"status": "evaluated", "delta": 0.4}}},
            {"sample_id": "unmapped", "modalities": {"audio": {"status": "evaluated", "delta": 0.9}}},
        ]
    }
    artifacts = {"segments": [{"sample_id": "second", "start_sec": 2.0, "end_sec": 3.0}]}
    timeline = extract_xai_timeline(xai, artifacts, {})
    assert timeline["audio"]["value"][0]["start"] == 2.0
    assert len(timeline["audio"]["value"]) == 1
    markup = build_xai_timeline_html(timeline, 0.0, 5.0)
    assert 'data-start="0.0"' in markup
    assert "data-studio-xai-zero" in markup
    assert "data-studio-xai-cursor" in markup
    assert "time (s)" in markup or ">2<" in markup


def test_xai_without_temporal_link_has_overview_only() -> None:
    xai = {"samples": [{"sample_id": "s", "baseline": 1.0, "modalities": {}}]}
    assert extract_xai_timeline(xai, {}, {}) == {}
    assert "No temporally linked" in build_xai_timeline_html({})


def test_run_inference_collects_existing_xai_step_artifact(tmp_path: Path) -> None:
    step_name = f"studio_xai_callback_{uuid4().hex}"

    class XAIModel(nn.Module):
        def forward(self, batch: Batch) -> ModelOutput:
            return ModelOutput(preds=sum(batch.inputs.values()))

    model = XAIModel().eval()
    batch = Batch(
        inputs={
            "audio": torch.tensor([[2.0]]),
            "video": torch.tensor([[3.0]]),
        },
        targets=torch.zeros(1, 1),
        meta={"sample_ids": ["sample-1"]},
    )

    class CallbackStep:
        on_prediction = None

        def run(self, context):
            output = model(batch)
            if self.on_prediction is not None:
                self.on_prediction(model, batch, output)
            context.set_artifact("predictions", {"score": output.preds.tolist()})
            return context

    @INFERENCE_STEPS.register(step_name)
    def build_callback_step():
        return CallbackStep()

    config_path = tmp_path / "inference.yaml"
    config_text = (
        "pipeline:\n  name: xai-test\nsteps:\n  - name: xai_step\n"
        "    params:\n      step:\n        name: " + step_name + "\n"
    )
    config_path.write_text(config_text, encoding="utf-8")
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")

    result = run_inference(input_path, config_path, device="cpu", work_dir_root=tmp_path)

    assert result["xai"]["samples"][0]["sample_id"] == "sample-1"
    assert result["xai"]["samples"][0]["modalities"]["audio"]["status"] == "evaluated"
    assert result["xai"]["samples"][0]["modalities"]["audio"]["delta"] == [2.0]
