"""Minimal Gradio interface for running registered inference pipelines."""

import json
import tempfile
from pathlib import Path
from typing import Any

from chimera_ml.inference import InferenceConfig, InferenceContext, build_inference_pipeline, resolve_inference_device


def discover_inference_configs(search_root: str | Path = "examples") -> list[str]:
    """Return inference YAML files below an examples directory."""
    root = Path(search_root)
    return sorted(str(path) for path in root.rglob("inference.yaml") if path.is_file()) if root.exists() else []


def json_safe(value: Any) -> Any:
    """Convert pipeline artifacts into a compact JSON-compatible view."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"type": type(value).__name__, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def run_inference(
    input_path: str | Path, config_path: str | Path, *, device: str = "auto", work_dir_root: str | Path | None = None
) -> dict[str, Any]:
    """Run one existing inference pipeline in a fresh working directory."""
    input_file = Path(input_path)
    config_file = Path(config_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input media file does not exist: {input_file}")
    if not config_file.is_file():
        raise FileNotFoundError(f"Inference config does not exist: {config_file}")

    work_dir = Path(tempfile.mkdtemp(prefix="chimera-studio-", dir=str(work_dir_root) if work_dir_root else None))
    config = InferenceConfig.from_yaml(str(config_file))
    context = InferenceContext(
        input_path=input_file,
        work_dir=work_dir,
        device=resolve_inference_device(device or config.runtime_device()),
        config=config.raw,
    )
    result = build_inference_pipeline(config).run(context)
    return {
        "predictions": json_safe(result.predictions),
        "artifacts": json_safe(result.artifacts),
        "work_dir": str(work_dir),
    }


def create_demo(config_paths: list[str] | None = None, *, search_root: str | Path = "examples") -> Any:
    """Build the Studio Gradio app; Gradio is imported only when called."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Studio requires Gradio. Install it with: pip install 'chimera-ml[studio]'") from exc

    choices = config_paths or discover_inference_configs(search_root)

    def execute(input_path: str | None, config_path: str | None, device: str) -> tuple[Any, str, str]:
        if not input_path:
            return None, "", "Select a video before starting inference."
        if not config_path:
            return None, "", "Select an inference YAML configuration."
        try:
            result = run_inference(input_path, config_path, device=device)
        except Exception as exc:
            return None, "", f"{type(exc).__name__}: {exc}"
        return result["predictions"], json.dumps(result["artifacts"], indent=2, ensure_ascii=False), ""

    with gr.Blocks(title="Chimera ML Studio") as demo:
        gr.Markdown("# Chimera ML Studio")
        with gr.Row():
            config = gr.Dropdown(choices=choices, value=choices[0] if choices else None, label="Inference YAML")
            device = gr.Dropdown(choices=["auto", "cpu", "cuda"], value="auto", label="Device")
        video = gr.Video(label="Video", type="filepath")
        run_button = gr.Button("Run inference", variant="primary")
        predictions = gr.JSON(label="Predictions")
        artifacts = gr.Code(label="Raw inference artifacts", language="json")
        errors = gr.Textbox(label="Errors", interactive=False)
        run_button.click(execute, inputs=[video, config, device], outputs=[predictions, artifacts, errors])
    return demo


def launch_studio(
    config_paths: list[str] | None = None,
    *,
    search_root: str | Path = "examples",
    host: str | None = None,
    port: int | None = None,
    share: bool = False,
) -> None:
    """Build and launch the Studio app."""
    demo = create_demo(config_paths, search_root=search_root)
    kwargs: dict[str, Any] = {"share": share}
    if host is not None:
        kwargs["server_name"] = host
    if port is not None:
        kwargs["server_port"] = port
    demo.launch(**kwargs)
