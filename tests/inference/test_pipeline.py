import json
from uuid import uuid4

import pytest

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference import InferenceConfig, InferenceContext, build_inference_pipeline, build_inference_step


class _OrderStep:
    def __init__(self, label: str) -> None:
        self.label = label

    def run(self, ctx: InferenceContext) -> InferenceContext:
        ctx.artifacts.setdefault("order", []).append(self.label)
        return ctx


def _make_ctx(tmp_path, *, artifacts=None) -> InferenceContext:
    return InferenceContext(
        input_path=tmp_path / "input.mp4",
        work_dir=tmp_path,
        device="cpu",
        config={},
        artifacts=artifacts or {},
    )


def test_inference_pipeline_executes_steps_in_config_order(tmp_path):
    first_name = f"test_infer_first_{uuid4().hex}"
    second_name = f"test_infer_second_{uuid4().hex}"

    @INFERENCE_STEPS.register(first_name)
    def _first_step(label: str):
        return _OrderStep(label)

    @INFERENCE_STEPS.register(second_name)
    def _second_step(label: str):
        return _OrderStep(label)

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"name": "demo"},
                "steps": [
                    {"name": first_name, "params": {"label": "first"}},
                    {"name": second_name, "params": {"label": "second"}},
                ],
            }
        )
    )

    ctx = pipeline.run(_make_ctx(tmp_path))

    assert ctx.artifacts["order"] == ["first", "second"]


def test_build_inference_step_resolves_registry_step_params(tmp_path):
    step_name = f"test_infer_step_{uuid4().hex}"
    seen: dict[str, object] = {}

    class _ConfiguredStep:
        def __init__(self, value: int) -> None:
            self.value = value

        def run(self, ctx: InferenceContext) -> InferenceContext:
            ctx.artifacts["value"] = self.value
            return ctx

    @INFERENCE_STEPS.register(step_name)
    def _factory(value: int):
        seen["value"] = value
        return _ConfiguredStep(value)

    step = build_inference_step({"name": step_name, "params": {"value": 7}})
    ctx = step.run(_make_ctx(tmp_path))

    assert seen["value"] == 7
    assert ctx.artifacts["value"] == 7


def test_build_inference_pipeline_rejects_non_mapping_step_config():
    cfg = InferenceConfig({"steps": ["not-a-mapping"]})

    with pytest.raises(TypeError, match="must be a mapping"):
        build_inference_pipeline(cfg)


def test_builtin_write_json_predictions_step_writes_json(tmp_path):
    output_path = tmp_path / "out.json"
    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"name": "demo"},
                "steps": [
                    {
                        "name": "write_json_predictions_step",
                        "params": {"output_path": str(output_path)},
                    }
                ],
            }
        )
    )
    ctx = pipeline.run(_make_ctx(tmp_path, artifacts={"predictions": [{"score": 0.5}]}))

    assert output_path.exists()
    assert ctx.get_artifact("output_path") == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "input": str(tmp_path / "input.mp4"),
        "predictions": [{"score": 0.5}],
    }


def test_builtin_write_json_predictions_step_uses_default_output_path(tmp_path):
    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"name": "demo"},
                "steps": [{"name": "write_json_predictions_step", "params": {}}],
            }
        )
    )
    ctx = pipeline.run(_make_ctx(tmp_path, artifacts={"predictions": [{"score": 0.5}]}))

    output_path = tmp_path / "input.json"
    assert output_path.exists()
    assert ctx.get_artifact("output_path") == output_path


def test_builtin_print_json_predictions_step_prints_json(tmp_path, capsys):
    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"name": "demo"},
                "steps": [{"name": "print_json_predictions_step", "params": {}}],
            }
        )
    )

    pipeline.run(_make_ctx(tmp_path, artifacts={"predictions": [{"score": 0.5}]}))

    out = capsys.readouterr().out
    assert '"input":' in out
    assert '"score": 0.5' in out
