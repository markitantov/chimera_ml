import json
import time
from uuid import uuid4

import pytest

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference import InferenceConfig, InferenceContext, build_inference_pipeline, build_inference_step
from chimera_ml.inference.pipeline import InferenceGraphNode, InferencePipeline


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
    first_name = f"test_inference_first_{uuid4().hex}"
    second_name = f"test_inference_second_{uuid4().hex}"

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


def test_inference_pipeline_prints_step_start_messages(tmp_path, capsys):
    first_name = f"test_inference_log_first_{uuid4().hex}"
    second_name = f"test_inference_log_second_{uuid4().hex}"

    @INFERENCE_STEPS.register(first_name)
    def _first_step():
        return _OrderStep("first")

    @INFERENCE_STEPS.register(second_name)
    def _second_step():
        return _OrderStep("second")

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"name": "demo"},
                "steps": [
                    {"name": first_name},
                    {"name": second_name},
                ],
            }
        )
    )

    pipeline.run(_make_ctx(tmp_path))

    out = capsys.readouterr().out
    assert f"[inference] Starting step '{first_name}' (_OrderStep)" in out
    assert f"[inference] Starting step '{second_name}' (_OrderStep)" in out


def test_build_inference_step_resolves_registry_step_params(tmp_path):
    step_name = f"test_inference_step_{uuid4().hex}"
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


def test_inference_pipeline_runs_independent_steps_in_parallel(tmp_path):
    first_name = f"test_inference_parallel_first_{uuid4().hex}"
    second_name = f"test_inference_parallel_second_{uuid4().hex}"
    join_name = f"test_inference_parallel_join_{uuid4().hex}"

    class _SleepAndStoreStep:
        def __init__(self, key: str, value: str, delay_sec: float = 0.2) -> None:
            self.key = key
            self.value = value
            self.delay_sec = delay_sec

        def run(self, ctx: InferenceContext) -> InferenceContext:
            time.sleep(self.delay_sec)
            ctx.set_artifact(self.key, self.value)
            return ctx

    class _JoinStep:
        def run(self, ctx: InferenceContext) -> InferenceContext:
            ctx.set_artifact(
                "joined",
                [ctx.get_artifact("audio_feature"), ctx.get_artifact("video_feature")],
            )
            return ctx

    @INFERENCE_STEPS.register(first_name)
    def _first_step():
        return _SleepAndStoreStep("audio_feature", "audio")

    @INFERENCE_STEPS.register(second_name)
    def _second_step():
        return _SleepAndStoreStep("video_feature", "video")

    @INFERENCE_STEPS.register(join_name)
    def _join_step():
        return _JoinStep()

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"name": "parallel-demo", "parallel": True},
                "steps": [
                    {"name": first_name},
                    {"name": second_name},
                    {"name": join_name, "after": [first_name, second_name]},
                ],
            }
        )
    )

    started_at = time.perf_counter()
    ctx = pipeline.run(_make_ctx(tmp_path))
    elapsed = time.perf_counter() - started_at

    assert ctx.artifacts["joined"] == ["audio", "video"]
    assert elapsed < 0.35


def test_inference_pipeline_keeps_sequential_order_without_parallel_flag(tmp_path):
    first_name = f"test_inference_seq_first_{uuid4().hex}"
    second_name = f"test_inference_seq_second_{uuid4().hex}"
    third_name = f"test_inference_seq_third_{uuid4().hex}"

    class _SleepAndStoreStep:
        def __init__(self, key: str, value: str, delay_sec: float = 0.2) -> None:
            self.key = key
            self.value = value
            self.delay_sec = delay_sec

        def run(self, ctx: InferenceContext) -> InferenceContext:
            time.sleep(self.delay_sec)
            ctx.set_artifact(self.key, self.value)
            return ctx

    @INFERENCE_STEPS.register(first_name)
    def _first_step():
        return _SleepAndStoreStep("audio_feature", "audio")

    @INFERENCE_STEPS.register(second_name)
    def _second_step():
        return _SleepAndStoreStep("video_feature", "video")

    @INFERENCE_STEPS.register(third_name)
    def _third_step():
        return _SleepAndStoreStep("joined", "done")

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"name": "sequential-demo"},
                "steps": [
                    {"name": first_name},
                    {"name": second_name},
                    {"name": third_name, "after": [first_name]},
                ],
            }
        )
    )

    started_at = time.perf_counter()
    ctx = pipeline.run(_make_ctx(tmp_path))
    elapsed = time.perf_counter() - started_at

    assert ctx.artifacts["audio_feature"] == "audio"
    assert ctx.artifacts["video_feature"] == "video"
    assert ctx.artifacts["joined"] == "done"
    assert elapsed >= 0.55


def test_inference_pipeline_waits_for_step_dependencies(tmp_path):
    produce_name = f"test_inference_produce_{uuid4().hex}"
    consume_name = f"test_inference_consume_{uuid4().hex}"

    class _ProduceStep:
        def run(self, ctx: InferenceContext) -> InferenceContext:
            ctx.set_artifact("audio_path", tmp_path / "input.wav")
            return ctx

    class _ConsumeStep:
        def run(self, ctx: InferenceContext) -> InferenceContext:
            audio_path = ctx.get_artifact("audio_path")
            if audio_path is None:
                raise AssertionError("dependency artifact is missing")
            ctx.set_artifact("vad_segments", [{"start": 0.0, "end": 1.0}])
            return ctx

    @INFERENCE_STEPS.register(produce_name)
    def _produce_step():
        return _ProduceStep()

    @INFERENCE_STEPS.register(consume_name)
    def _consume_step():
        return _ConsumeStep()

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"name": "dependency-demo", "parallel": True},
                "steps": [
                    {"name": produce_name},
                    {
                        "name": consume_name,
                        "after": [produce_name],
                    },
                ],
            }
        )
    )

    ctx = pipeline.run(_make_ctx(tmp_path))

    assert ctx.artifacts["audio_path"] == tmp_path / "input.wav"
    assert ctx.artifacts["vad_segments"] == [{"start": 0.0, "end": 1.0}]


def test_inference_pipeline_rejects_unknown_dependency():
    with pytest.raises(ValueError, match="unknown step"):
        InferencePipeline(
            nodes=[
                InferenceGraphNode(
                    node_id="step_a",
                    step=_OrderStep("ok"),
                    after=("missing",),
                )
            ]
        )


def test_build_inference_pipeline_rejects_cycle(tmp_path):
    first_name = f"test_inference_cycle_first_{uuid4().hex}"
    second_name = f"test_inference_cycle_second_{uuid4().hex}"

    @INFERENCE_STEPS.register(first_name)
    def _first():
        return _OrderStep("first")

    @INFERENCE_STEPS.register(second_name)
    def _second():
        return _OrderStep("second")

    with pytest.raises(ValueError, match="cycle"):
        build_inference_pipeline(
            InferenceConfig(
                {
                    "pipeline": {"parallel": True},
                    "steps": [
                        {"id": "first", "name": first_name, "after": ["second"]},
                        {"id": "second", "name": second_name, "after": ["first"]},
                    ],
                }
            )
        )


def test_inference_pipeline_rejects_duplicate_artifact_writes_in_graph_mode(tmp_path):
    first_name = f"test_inference_dup_artifact_first_{uuid4().hex}"
    second_name = f"test_inference_dup_artifact_second_{uuid4().hex}"
    join_name = f"test_inference_dup_artifact_join_{uuid4().hex}"

    @INFERENCE_STEPS.register(first_name)
    def _first():
        class _FirstStep:
            def run(self, ctx: InferenceContext) -> InferenceContext:
                ctx.set_artifact("shared", "first")
                return ctx

        return _FirstStep()

    @INFERENCE_STEPS.register(second_name)
    def _second():
        class _SecondStep:
            def run(self, ctx: InferenceContext) -> InferenceContext:
                ctx.set_artifact("shared", "second")
                return ctx

        return _SecondStep()

    @INFERENCE_STEPS.register(join_name)
    def _join():
        return _OrderStep("join")

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"parallel": True},
                "steps": [
                    {"name": first_name},
                    {"name": second_name},
                    {"name": join_name, "after": [first_name, second_name]},
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="already written"):
        pipeline.run(_make_ctx(tmp_path))


def test_inference_pipeline_allows_overwriting_artifact_from_dependency_chain(tmp_path):
    first_name = f"test_inference_overwrite_first_{uuid4().hex}"
    second_name = f"test_inference_overwrite_second_{uuid4().hex}"

    @INFERENCE_STEPS.register(first_name)
    def _first():
        class _FirstStep:
            def run(self, ctx: InferenceContext) -> InferenceContext:
                ctx.set_artifact("shared", [1, 2, 3])
                return ctx

        return _FirstStep()

    @INFERENCE_STEPS.register(second_name)
    def _second():
        class _SecondStep:
            def run(self, ctx: InferenceContext) -> InferenceContext:
                shared = list(ctx.get_artifact("shared", []))
                ctx.set_artifact("shared", [item * 10 for item in shared])
                return ctx

        return _SecondStep()

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"parallel": True},
                "steps": [
                    {"name": first_name},
                    {"name": second_name, "after": [first_name]},
                ],
            }
        )
    )

    ctx = pipeline.run(_make_ctx(tmp_path))

    assert ctx.get_artifact("shared") == [10, 20, 30]


def test_inference_pipeline_rejects_parallel_in_place_aliasing_updates(tmp_path):
    first_name = f"test_inference_alias_first_{uuid4().hex}"
    second_name = f"test_inference_alias_second_{uuid4().hex}"
    join_name = f"test_inference_alias_join_{uuid4().hex}"

    @INFERENCE_STEPS.register(first_name)
    def _first():
        class _FirstStep:
            def run(self, ctx: InferenceContext) -> InferenceContext:
                ctx.get_artifact("shared").append("first")
                return ctx

        return _FirstStep()

    @INFERENCE_STEPS.register(second_name)
    def _second():
        class _SecondStep:
            def run(self, ctx: InferenceContext) -> InferenceContext:
                ctx.get_artifact("shared").append("second")
                return ctx

        return _SecondStep()

    @INFERENCE_STEPS.register(join_name)
    def _join():
        return _OrderStep("join")

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"parallel": True},
                "steps": [
                    {"name": first_name},
                    {"name": second_name},
                    {"name": join_name, "after": [first_name, second_name]},
                ],
            }
        )
    )

    shared: list[str] = []
    ctx = _make_ctx(tmp_path, artifacts={"shared": shared})

    with pytest.raises(ValueError, match="already written"):
        pipeline.run(ctx)

    assert shared == []


def test_build_inference_pipeline_uses_step_name_as_default_node_id(tmp_path):
    first_name = f"test_inference_default_id_first_{uuid4().hex}"
    second_name = f"test_inference_default_id_second_{uuid4().hex}"

    class _StoreStep:
        def __init__(self, key: str, value: str) -> None:
            self.key = key
            self.value = value

        def run(self, ctx: InferenceContext) -> InferenceContext:
            ctx.set_artifact(self.key, self.value)
            return ctx

    @INFERENCE_STEPS.register(first_name)
    def _first():
        return _StoreStep("first_done", "first")

    @INFERENCE_STEPS.register(second_name)
    def _second():
        return _StoreStep("second_done", "second")

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"parallel": True},
                "steps": [
                    {"name": first_name},
                    {"name": second_name, "after": [first_name]},
                ],
            }
        )
    )

    ctx = pipeline.run(_make_ctx(tmp_path))

    assert ctx.get_artifact("first_done") == "first"
    assert ctx.get_artifact("second_done") == "second"


def test_inference_pipeline_propagates_exception_from_parallel_step(tmp_path):
    failing_name = f"test_inference_failing_{uuid4().hex}"
    slow_name = f"test_inference_slow_{uuid4().hex}"

    @INFERENCE_STEPS.register(failing_name)
    def _failing():
        class _FailingStep:
            def run(self, ctx: InferenceContext) -> InferenceContext:
                raise RuntimeError("boom")

        return _FailingStep()

    @INFERENCE_STEPS.register(slow_name)
    def _slow():
        class _SlowStep:
            def run(self, ctx: InferenceContext) -> InferenceContext:
                time.sleep(0.2)
                ctx.set_artifact("slow", True)
                return ctx

        return _SlowStep()

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"parallel": True},
                "steps": [
                    {"name": failing_name},
                    {"name": slow_name},
                ],
            }
        )
    )

    ctx = _make_ctx(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run(ctx)

    assert ctx.artifacts == {}


def test_inference_pipeline_rolls_back_merged_artifacts_when_later_step_fails(tmp_path):
    first_name = f"test_inference_rollback_first_{uuid4().hex}"
    second_name = f"test_inference_rollback_second_{uuid4().hex}"

    @INFERENCE_STEPS.register(first_name)
    def _first():
        class _FirstStep:
            def run(self, ctx: InferenceContext) -> InferenceContext:
                ctx.set_artifact("done", 1)
                return ctx

        return _FirstStep()

    @INFERENCE_STEPS.register(second_name)
    def _second():
        class _SecondStep:
            def run(self, ctx: InferenceContext) -> InferenceContext:
                raise RuntimeError("late failure")

        return _SecondStep()

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "pipeline": {"parallel": True},
                "steps": [
                    {"name": first_name},
                    {"name": second_name, "after": [first_name]},
                ],
            }
        )
    )

    ctx = _make_ctx(tmp_path, artifacts={"existing": 42})
    with pytest.raises(RuntimeError, match="late failure"):
        pipeline.run(ctx)

    assert ctx.artifacts == {"existing": 42}


def test_build_inference_pipeline_requires_explicit_id_for_duplicate_step_names(tmp_path):
    step_name = f"test_inference_duplicate_name_{uuid4().hex}"
    join_name = f"test_inference_duplicate_name_join_{uuid4().hex}"

    @INFERENCE_STEPS.register(step_name)
    def _step(label: str):
        return _OrderStep(label)

    @INFERENCE_STEPS.register(join_name)
    def _join():
        return _OrderStep("join")

    with pytest.raises(ValueError, match="Duplicate node ids"):
        build_inference_pipeline(
            InferenceConfig(
                {
                    "pipeline": {"parallel": True},
                    "steps": [
                        {"name": step_name, "params": {"label": "first"}},
                        {"name": step_name, "params": {"label": "second"}},
                        {"name": join_name, "after": [step_name]},
                    ],
                }
            )
        )


def test_build_inference_pipeline_rejects_non_boolean_global_parallel(tmp_path):
    step_name = f"test_inference_parallel_type_{uuid4().hex}"

    @INFERENCE_STEPS.register(step_name)
    def _step():
        return _OrderStep("ok")

    with pytest.raises(TypeError, match=r"pipeline\.parallel"):
        build_inference_pipeline(
            InferenceConfig(
                {
                    "pipeline": {"parallel": "yes"},
                    "steps": [
                        {"name": step_name},
                    ],
                }
            )
        )


def test_build_inference_pipeline_warns_when_parallel_mode_has_no_after(tmp_path):
    first_name = f"test_inference_warn_parallel_first_{uuid4().hex}"
    second_name = f"test_inference_warn_parallel_second_{uuid4().hex}"

    @INFERENCE_STEPS.register(first_name)
    def _first():
        return _OrderStep("first")

    @INFERENCE_STEPS.register(second_name)
    def _second():
        return _OrderStep("second")

    with pytest.warns(UserWarning, match="no step defines 'after'"):
        build_inference_pipeline(
            InferenceConfig(
                {
                    "pipeline": {"parallel": True},
                    "steps": [
                        {"name": first_name},
                        {"name": second_name},
                    ],
                }
            )
        )
