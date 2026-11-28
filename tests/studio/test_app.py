from pathlib import Path
from uuid import uuid4

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.studio.app import discover_inference_configs, run_inference


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
