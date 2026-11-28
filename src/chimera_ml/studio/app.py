"""Gradio application for Chimera ML Studio."""

import json
from pathlib import Path
from typing import Any

from .inference import discover_inference_configs, json_safe, run_inference
from .predictions import build_prediction_overview, extract_timeline
from .timeline import build_timeline_html, build_xai_timeline_html, load_timeline_js
from .xai import extract_xai_timeline, render_xai_overview

__all__ = [
    "build_prediction_overview",
    "build_timeline_html",
    "build_xai_timeline_html",
    "create_demo",
    "discover_inference_configs",
    "extract_timeline",
    "extract_xai_timeline",
    "json_safe",
    "launch_studio",
    "render_xai_overview",
    "run_inference",
    "timeline_sync_js",
]


def timeline_sync_js() -> str:
    """Return the packaged browser synchronization script."""
    return load_timeline_js()


def create_demo(config_paths: list[str] | None = None, *, search_root: str | Path = "examples") -> Any:
    """Build the Studio Gradio app; Gradio is imported only when called."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Studio requires Gradio. Install it with: pip install 'chimera-ml[studio]'") from exc

    choices = config_paths or discover_inference_configs(search_root)

    def execute(
        input_path: str | None, config_path: str | None, device: str
    ) -> tuple[Any, Any, str, str, str, str, str]:
        if not input_path:
            return None, None, "", "", "", "", "Select a video before starting inference."
        if not config_path:
            return None, None, "", "", "", "", "Select an inference YAML configuration."
        try:
            result = run_inference(input_path, config_path, device=device)
        except Exception as exc:
            return None, None, "", "", "", "", f"{type(exc).__name__}: {exc}"
        timeline_end = result.get("timeline_duration")
        if timeline_end is None:
            timeline_end = max(
                (
                    point["end"]
                    for series in result["timeline"].values()
                    for points in series.values()
                    for point in points
                ),
                default=0.0,
            )
        return (
            result["overview"],
            result["predictions"],
            build_timeline_html(result["timeline"], result.get("timeline_duration")),
            render_xai_overview(result.get("xai")),
            build_xai_timeline_html(result["xai_timeline"], 0.0, timeline_end),
            json.dumps(result["artifacts"], indent=2, ensure_ascii=False),
            "",
        )

    with gr.Blocks(title="Chimera ML Studio") as demo:
        gr.Markdown("# Chimera ML Studio")
        with gr.Row():
            config = gr.Dropdown(choices=choices, value=choices[0] if choices else None, label="Inference YAML")
            device = gr.Dropdown(choices=["auto", "cpu", "cuda"], value="auto", label="Device")
        video = gr.Video(label="Video", elem_id="studio-video")
        run_button = gr.Button("Run inference", variant="primary")
        overview = gr.JSON(label="Prediction overview")
        raw_predictions = gr.JSON(label="Raw predictions")
        timeline = gr.HTML(value=build_timeline_html({}), elem_id="studio-timeline")
        xai_overview = gr.HTML(value=render_xai_overview(None), label="XAI overview")
        xai_timeline = gr.HTML(value=build_xai_timeline_html({}), elem_id="studio-xai-timeline")
        artifacts = gr.Code(label="Raw inference artifacts", language="json")
        errors = gr.Textbox(label="Errors", interactive=False)
        run_button.click(
            execute,
            inputs=[video, config, device],
            outputs=[overview, raw_predictions, timeline, xai_overview, xai_timeline, artifacts, errors],
        )
        run_button.click(None, js=timeline_sync_js())
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
