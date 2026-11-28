"""Minimal Gradio interface for running registered inference pipelines."""

import html
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from chimera_ml.inference import InferenceConfig, InferenceContext, build_inference_pipeline, resolve_inference_device


def discover_inference_configs(search_root: str | Path = "examples") -> list[str]:
    """Return inference YAML files below an examples directory."""
    root = Path(search_root)
    return sorted(str(path) for path in root.rglob("inference.yaml") if path.is_file()) if root.exists() else []


def json_safe(value: Any) -> Any:
    """Convert pipeline artifacts to JSON without dropping numeric tensors."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
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
    predictions = json_safe(result.predictions)
    artifacts = json_safe(result.artifacts)
    return {
        "predictions": predictions,
        "artifacts": artifacts,
        "overview": build_prediction_overview(predictions),
        "timeline": extract_timeline(artifacts, predictions),
        "work_dir": str(work_dir),
    }


def build_prediction_overview(predictions: Any) -> Any:
    """Build a task-agnostic, display-friendly view of prediction outputs."""
    if isinstance(predictions, dict):
        overview: dict[str, Any] = {}
        for key, value in predictions.items():
            if key in {"segments", "windows", "timeline"}:
                continue
            if isinstance(value, dict) and "probabilities" in value:
                item = {"probabilities": value["probabilities"]}
                for field in ("label", "class", "class_index", "value"):
                    if field in value:
                        item[field] = value[field]
                overview[str(key)] = item
            elif isinstance(value, (dict, list)):
                nested = build_prediction_overview(value)
                if nested not in ({}, []):
                    overview[str(key)] = nested
            else:
                overview[str(key)] = value
        return overview
    if isinstance(predictions, list):
        return [build_prediction_overview(item) for item in predictions]
    return predictions


def extract_timeline(artifacts: Any, predictions: Any) -> dict[str, dict[str, list[dict[str, float]]]]:
    """Extract real interval-valued series from inference results.

    The function consumes common plugin artifacts such as ``timeline``,
    ``modality_predictions`` and ``segments``. It never fills gaps or creates
    samples between supplied intervals.
    """
    sources: list[tuple[str, Any]] = []
    if isinstance(artifacts, dict):
        for key in ("timeline", "temporal_predictions", "modality_predictions", "fusion_predictions"):
            if key in artifacts:
                sources.extend(iter_sources(key, artifacts[key]))
    if isinstance(predictions, dict):
        if isinstance(predictions.get("segments"), list):
            sources.append(("fusion", predictions["segments"]))
        if isinstance(predictions.get("windows"), list):
            sources.append(("fusion", predictions["windows"]))
        for key, value in predictions.items():
            if key not in {"segments", "windows"} and isinstance(value, list):
                sources.append((str(key), value))

    result: dict[str, dict[str, list[dict[str, float]]]] = {}
    for source_name, records in sources:
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            start = record.get("start_sec", record.get("start"))
            end = record.get("end_sec", record.get("end"))
            if start is None:
                start = record.get("time_sec", record.get("timestamp"))
                end = start
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            for output, value in numeric_outputs(record):
                result.setdefault(source_name, {}).setdefault(output, []).append(
                    {"start": float(start), "end": float(end), "value": float(value)}
                )
    return result


def iter_sources(name: str, value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for source, records in value.items():
            if isinstance(records, list):
                yield str(source), records
    elif isinstance(value, list):
        yield name.removesuffix("_predictions"), value


def numeric_outputs(record: dict[str, Any], prefix: str = "") -> Iterable[tuple[str, float]]:
    for key, value in record.items():
        if key in {"start", "end", "start_sec", "end_sec", "time_sec", "timestamp", "index", "weight"}:
            continue
        output = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            if isinstance(value.get("probabilities"), dict):
                yield from numeric_outputs(value["probabilities"], output)
            else:
                yield from numeric_outputs(value, output)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield output, float(value)


def build_timeline_html(timeline: dict[str, dict[str, list[dict[str, float]]]]) -> str:
    """Render interval series and browser-side controls as an SVG timeline."""
    if not timeline:
        return '<div class="studio-empty-timeline">No temporal predictions available.</div>'
    all_points = [point for series in timeline.values() for points in series.values() for point in points]
    minimum = min(point["start"] for point in all_points)
    maximum = max(point["end"] for point in all_points)
    span = max(maximum - minimum, 1e-6)
    width, height = 960, 360
    output_names = sorted({output for series in timeline.values() for output in series})
    parts = [
        '<div class="studio-timeline-controls">',
        "<label>Output <select data-studio-output>",
        *[f'<option value="{html.escape(name, quote=True)}">{html.escape(name)}</option>' for name in output_names],
        "</select></label>",
    ]
    for source in sorted(timeline):
        escaped = html.escape(source, quote=True)
        parts.append(f'<label><input type="checkbox" data-studio-source="{escaped}" checked> {escaped}</label>')
    parts.append("</div>")
    parts.append(
        f'<svg data-studio-timeline viewBox="0 0 {width} {height}" role="img" '
        f'data-start="{minimum}" data-end="{maximum}"><line data-studio-cursor x1="0" x2="0" y1="0" y2="{height}" />'
    )
    for source, series in sorted(timeline.items()):
        for output, points in sorted(series.items()):
            values = [point["value"] for point in points]
            low, high = min(values), max(values)
            value_span = max(high - low, 1e-6)
            path_parts: list[str] = []
            for point in points:
                x1 = (point["start"] - minimum) / span * width
                x2 = (point["end"] - minimum) / span * width
                y = height - (point["value"] - low) / value_span * (height - 30) - 15
                path_parts.append(f"M{x1:.3f},{y:.3f} L{x2:.3f},{y:.3f}")
            data = html.escape(json.dumps(points, separators=(",", ":")), quote=True)
            parts.append(
                f'<path data-studio-series data-source="{html.escape(source, quote=True)}" '
                f'data-output="{html.escape(output, quote=True)}" data-points="{data}" '
                f'd="{" ".join(path_parts)}" fill="none" stroke-width="2" />'
            )
    parts.append("</svg>")
    parts.append("<script>window.__chimeraStudioTimeline = true;</script>")
    return "".join(parts)


def timeline_sync_js() -> str:
    """Return the minimal browser-side video/timeline synchronization script."""
    return r"""
    () => {
      const root = document.querySelector('#studio-timeline');
      const videoRoot = document.querySelector('#studio-video');
      if (!root || !videoRoot) return;
      const bind = () => {
        const video = videoRoot.querySelector('video');
        const svg = root.querySelector('[data-studio-timeline]');
        if (!video || !svg || svg.dataset.bound === 'true') return;
        svg.dataset.bound = 'true';
        const cursor = svg.querySelector('[data-studio-cursor]');
        const updateCursor = () => {
          const start = Number(svg.dataset.start), end = Number(svg.dataset.end);
          const time = Number(video.currentTime || 0);
          const x = Math.max(0, Math.min(1, (time - start) / Math.max(end - start, 1e-9))) * 960;
          cursor.setAttribute('x1', String(x)); cursor.setAttribute('x2', String(x));
        };
        video.addEventListener('timeupdate', updateCursor);
        video.addEventListener('seeking', updateCursor);
        svg.addEventListener('click', (event) => {
          const point = svg.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
          const transformed = point.matrixTransform(svg.getScreenCTM().inverse());
          const start = Number(svg.dataset.start), end = Number(svg.dataset.end);
          video.currentTime = start + (transformed.x / 960) * (end - start);
        });
        root.querySelector('[data-studio-output]')?.addEventListener('change', (event) => {
          root.querySelectorAll('[data-studio-series]').forEach((path) => {
            path.style.display = path.dataset.output === event.target.value ? '' : 'none';
          });
        });
        root.querySelectorAll('[data-studio-source]').forEach((box) => box.addEventListener('change', () => {
          const selectedOutput = root.querySelector('[data-studio-output]').value;
          const selector = `[data-studio-series][data-source="${box.dataset.studioSource}"]`;
          root.querySelectorAll(selector).forEach((path) => {
            path.style.display = box.checked && path.dataset.output === selectedOutput ? '' : 'none';
          });
        }));
        updateCursor();
      };
      bind();
      new MutationObserver(bind).observe(root, {childList: true, subtree: true});
    }
    """


def create_demo(config_paths: list[str] | None = None, *, search_root: str | Path = "examples") -> Any:
    """Build the Studio Gradio app; Gradio is imported only when called."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Studio requires Gradio. Install it with: pip install 'chimera-ml[studio]'") from exc

    choices = config_paths or discover_inference_configs(search_root)

    def execute(input_path: str | None, config_path: str | None, device: str) -> tuple[Any, Any, str, str, str]:
        if not input_path:
            return None, None, "", "", "Select a video before starting inference."
        if not config_path:
            return None, None, "", "", "Select an inference YAML configuration."
        try:
            result = run_inference(input_path, config_path, device=device)
        except Exception as exc:
            return None, None, "", "", f"{type(exc).__name__}: {exc}"
        return (
            result["overview"],
            result["predictions"],
            build_timeline_html(result["timeline"]),
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
        artifacts = gr.Code(label="Raw inference artifacts", language="json")
        errors = gr.Textbox(label="Errors", interactive=False)
        run_button.click(
            execute, inputs=[video, config, device], outputs=[overview, raw_predictions, timeline, artifacts, errors]
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
