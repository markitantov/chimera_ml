import json
from pathlib import Path
from uuid import uuid4

import pytest
import torch
from torch import nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.studio.app import (
    build_prediction_overview,
    build_timeline_html,
    build_xai_timeline_html,
    create_demo,
    discover_inference_configs,
    extract_timeline,
    extract_xai_timeline,
    json_safe,
    render_xai_overview,
    run_inference,
    timeline_sync_js,
)


def test_discover_inference_configs_returns_sorted_files(tmp_path: Path) -> None:
    (tmp_path / "b" / "configs").mkdir(parents=True)
    (tmp_path / "a" / "configs").mkdir(parents=True)
    (tmp_path / "b" / "configs" / "inference.yaml").write_text("pipeline: {}\n", encoding="utf-8")
    (tmp_path / "a" / "configs" / "inference.yaml").write_text("pipeline: {}\n", encoding="utf-8")
    assert discover_inference_configs(tmp_path) == [
        str(tmp_path / "a" / "configs" / "inference.yaml"),
        str(tmp_path / "b" / "configs" / "inference.yaml"),
    ]


def test_run_inference_uses_isolated_work_directories(tmp_path: Path) -> None:
    step_name = f"studio_test_step_{uuid4().hex}"

    class ProduceResultStep:
        def run(self, ctx):
            ctx.set_artifact("predictions", {"ok": True})
            ctx.set_artifact("work_dir_seen", str(ctx.work_dir))
            return ctx

    @INFERENCE_STEPS.register(step_name)
    def build_step():
        return ProduceResultStep()

    config_path = tmp_path / "inference.yaml"
    config_path.write_text(f"pipeline:\n  name: test\nsteps:\n  - name: {step_name}\n", encoding="utf-8")
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    work_root = tmp_path / "runs"
    work_root.mkdir()
    first = run_inference(input_path, config_path, device="cpu", work_dir_root=work_root)
    second = run_inference(input_path, config_path, device="cpu", work_dir_root=work_root)
    assert first["predictions"] == {"ok": True}
    assert first["work_dir"] != second["work_dir"]
    assert Path(first["work_dir"]).parent == work_root
    assert first["artifacts"]["work_dir_seen"] == first["work_dir"]


def test_prediction_overview_is_task_agnostic() -> None:
    overview = build_prediction_overview(
        {
            "emotion": {"label": "happy", "class_index": 1, "probabilities": {"sad": 0.1, "happy": 0.9}},
            "age": {"value": 32.5},
            "text": "hello",
            "segments": [{"start_sec": 0.0}],
        }
    )

    assert overview["emotion"]["label"] == "happy"
    assert overview["emotion"]["probabilities"]["happy"] == 0.9
    assert overview["age"]["value"] == 32.5
    assert "segments" not in overview


def test_json_safe_preserves_numeric_tensor_values() -> None:
    assert json_safe(torch.tensor([0.25, 0.75])) == [0.25, 0.75]


def test_timeline_keeps_real_intervals_and_gaps() -> None:
    artifacts = {
        "modality_predictions": {
            "audio": [
                {"start_sec": 0.0, "end_sec": 1.0, "score": 0.2},
                {"start_sec": 2.0, "end_sec": 3.5, "score": 0.8},
            ],
            "video": [{"start_sec": 0.0, "end_sec": 1.5, "score": 0.4}],
        }
    }
    predictions = {"segments": [{"start_sec": 0.5, "end_sec": 1.25, "result": {"probabilities": {"x": 0.7}}}]}

    timeline = extract_timeline(artifacts, predictions)
    assert len(timeline["audio"]["score"]) == 2
    assert timeline["audio"]["score"][1]["start"] == 2.0
    assert timeline["fusion"]["result.x"][0]["value"] == 0.7
    markup = build_timeline_html(timeline)
    assert 'data-start="0.0"' in markup
    assert 'data-source="audio"' in markup
    assert 'data-source="video"' in markup
    assert 'data-output="score"' in markup


def test_timeline_without_temporal_predictions_is_explicit() -> None:
    assert extract_timeline({}, {"label": "happy"}) == {}
    assert "No temporal predictions" in build_timeline_html({})


def test_create_demo_registers_browser_sync() -> None:
    pytest.importorskip("gradio")
    demo = create_demo(config_paths=["example.yaml"])
    config = demo.get_config_file()
    serialized = json.dumps(config)
    assert "studio-timeline" in serialized
    assert "currentTime" in serialized
    assert "currentTime" in timeline_sync_js()
    assert "timeupdate" in timeline_sync_js()
    assert "studio-xai-timeline" in timeline_sync_js()
    assert "plotLeft" in timeline_sync_js()


def test_timeline_uses_common_output_scale_and_video_duration() -> None:
    timeline = {
        "audio": {"probability": [{"start": 0.0, "end": 1.0, "value": 0.2}]},
        "video": {"probability": [{"start": 2.0, "end": 2.0, "value": 0.8}]},
    }
    markup = build_timeline_html(timeline, duration_sec=5.0)
    assert 'data-start="0.0"' in markup
    assert 'data-end="5.0"' in markup
    assert 'data-plot-left="70"' in markup
    assert 'data-plot-width="865"' in markup
    assert ">0</text>" in markup
    assert ">1</text>" in markup
    assert 'stroke="#2563eb"' in markup
    assert 'stroke="#dc2626"' in markup
    assert "data-studio-cursor" in markup
    assert "cx=" in markup
    assert "time (s)" in markup


def test_timeline_excludes_service_numeric_fields() -> None:
    timeline = extract_timeline(
        {},
        {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "duration_sec": 1.0,
                    "num_frames": 20,
                    "has_faces": True,
                    "emotion": {"probabilities": {"happy": 0.7}},
                }
            ]
        },
    )
    assert "emotion.happy" in timeline["fusion"]
    assert all("duration" not in key and "num_" not in key for key in timeline["fusion"])


def test_timeline_sync_tears_down_by_reusing_single_observer() -> None:
    script = timeline_sync_js()
    assert "root.__chimeraStudioSync" in script
    assert "if (!state.observer)" in script
    assert "data-studio-source]:checked" in script
    assert "Math.min(limit, target)" in script
    assert "data-studio-axis-output" in script


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
