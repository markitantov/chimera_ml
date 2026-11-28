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
    try:
        import cv2

        capture = cv2.VideoCapture(str(input_file))
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_rate = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        duration_sec = frame_count / frame_rate if frame_count > 0 and frame_rate > 0 else None
    except (ImportError, OSError):
        duration_sec = None

    predictions = json_safe(result.predictions)
    artifacts = json_safe(result.artifacts)
    return {
        "predictions": predictions,
        "artifacts": artifacts,
        "overview": build_prediction_overview(predictions),
        "timeline": extract_timeline(artifacts, predictions),
        "timeline_duration": duration_sec,
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
            for output, value, probability in numeric_outputs(record):
                result.setdefault(source_name, {}).setdefault(output, []).append(
                    {"start": float(start), "end": float(end), "value": float(value), "probability": probability}
                )
    return result


def iter_sources(name: str, value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for source, records in value.items():
            if isinstance(records, list):
                yield str(source), records
    elif isinstance(value, list):
        yield name.removesuffix("_predictions"), value


def numeric_outputs(
    record: dict[str, Any], prefix: str = "", probability: bool = False
) -> Iterable[tuple[str, float, bool]]:
    for key, value in record.items():
        if key in {
            "start",
            "end",
            "start_sec",
            "end_sec",
            "time_sec",
            "timestamp",
            "index",
            "weight",
            "duration_sec",
            "speech_duration_sec",
            "num_windows",
            "num_faces",
            "num_frames",
            "source_frame_index",
            "class_index",
        } or key.startswith("has_"):
            continue
        output = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            if isinstance(value.get("probabilities"), dict):
                yield from numeric_outputs(value["probabilities"], output, True)
            else:
                yield from numeric_outputs(value, output)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield output, float(value), probability


def build_timeline_html(
    timeline: dict[str, dict[str, list[dict[str, float]]]], duration_sec: float | None = None
) -> str:
    """Render interval series and browser-side controls as an SVG timeline."""
    if not timeline:
        return '<div class="studio-empty-timeline">No temporal predictions available.</div>'
    all_points = [point for series in timeline.values() for points in series.values() for point in points]
    minimum = min(point["start"] for point in all_points)
    maximum = max(point["end"] for point in all_points)
    if duration_sec is not None and duration_sec > maximum:
        maximum = duration_sec
    span = max(maximum - minimum, 1e-6)
    width, height = 960, 420
    left, right, top, bottom = 70, 25, 30, 55
    plot_width, plot_height = width - left - right, height - top - bottom
    output_names = sorted({output for series in timeline.values() for output in series})
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2"]
    source_colors = {source: colors[index % len(colors)] for index, source in enumerate(sorted(timeline))}
    output_ranges: dict[str, tuple[float, float]] = {}
    for output in output_names:
        output_points = [
            point
            for series in timeline.values()
            for name, points in series.items()
            if name == output
            for point in points
        ]
        values = [point["value"] for point in output_points]
        is_probability = (
            any(point.get("probability", False) for point in output_points)
            or "probabil" in output
            or output.endswith(("_prob", "_probs"))
        )
        low, high = (0.0, 1.0) if is_probability else (min(values), max(values))
        if high <= low:
            padding = max(abs(low) * 0.05, 1.0)
            low, high = low - padding, high + padding
        output_ranges[output] = (low, high)
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
        f'data-start="{minimum}" data-end="{maximum}" data-video-duration="{duration_sec or maximum}">'
    )
    for output in output_names:
        low, high = output_ranges[output]
        display = "" if output == output_names[0] else "none"
        escaped_output = html.escape(output, quote=True)
        parts.append(f'<g data-studio-axis-output="{escaped_output}" style="display:{display}">')
        parts.append(
            f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" stroke="#374151" />'
            f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" stroke="#374151" />'
            f'<text x="12" y="{top - 8}" font-size="12">{html.escape(output)}</text>'
        )
        for tick in range(6):
            value = low + (high - low) * tick / 5
            y = height - bottom - plot_height * tick / 5
            parts.append(
                f'<line x1="{left - 4}" x2="{left}" y1="{y:.3f}" y2="{y:.3f}" stroke="#374151" />'
                f'<text x="{left - 8}" y="{y + 4:.3f}" text-anchor="end" font-size="11">{value:.3g}</text>'
            )
        parts.append(f'<text x="{(left + width - right) / 2}" y="{height - 8}" text-anchor="middle">time (s)</text>')
        parts.append("</g>")
    parts.append(
        f'<line data-studio-cursor x1="{left}" x2="{left}" y1="{top}" '
        f'y2="{height - bottom}" stroke="#111827" stroke-width="2" />'
    )
    for source, series in sorted(timeline.items()):
        for output, points in sorted(series.items()):
            low, high = output_ranges[output]
            path_parts: list[str] = []
            point_parts: list[str] = []
            for point in points:
                x1 = left + (point["start"] - minimum) / span * plot_width
                x2 = left + (point["end"] - minimum) / span * plot_width
                y = height - bottom - (point["value"] - low) / max(high - low, 1e-9) * plot_height
                if point["start"] == point["end"]:
                    point_parts.append(f'<circle cx="{x1:.3f}" cy="{y:.3f}" r="4" />')
                else:
                    path_parts.append(f"M{x1:.3f},{y:.3f} L{x2:.3f},{y:.3f}")
            data = html.escape(json.dumps(points, separators=(",", ":")), quote=True)
            source_attr = html.escape(source, quote=True)
            output_attr = html.escape(output, quote=True)
            parts.append(
                f'<path data-studio-series data-source="{source_attr}" data-output="{output_attr}" '
                f'data-points="{data}" d="{" ".join(path_parts)}" fill="none" '
                f'stroke="{source_colors[source]}" stroke-width="2" />'
            )
            if point_parts:
                parts.append(
                    f'<g data-studio-series data-source="{source_attr}" data-output="{output_attr}" '
                    f'fill="{source_colors[source]}">' + "".join(point_parts) + "</g>"
                )
    parts.append('<g data-studio-legend transform="translate(70,15)">')
    for index, (source, color) in enumerate(source_colors.items()):
        x = index * 145
        parts.append(
            f'<rect x="{x}" y="-10" width="12" height="12" fill="{color}" />'
            f'<text x="{x + 17}" y="0">{html.escape(source)}</text>'
        )
    parts.append("</g></svg>")
    parts.append("<script>window.__chimeraStudioTimeline = true;</script>")
    return "".join(parts)


def timeline_sync_js() -> str:
    """Return browser-side video/timeline synchronization with teardown-safe binding."""
    return r"""
    () => {
      const root = document.querySelector('#studio-timeline');
      const videoRoot = document.querySelector('#studio-video');
      if (!root || !videoRoot) return;
      const state = root.__chimeraStudioSync || (root.__chimeraStudioSync = {});
      const applyVisibility = () => {
        const output = root.querySelector('[data-studio-output]')?.value;
        const active = new Set([...root.querySelectorAll('[data-studio-source]:checked')].map(
          (box) => box.dataset.studioSource
        ));
        root.querySelectorAll('[data-studio-series]').forEach((series) => {
          series.style.display = series.dataset.output === output && active.has(series.dataset.source) ? '' : 'none';
        });
        root.querySelectorAll('[data-studio-axis-output]').forEach((axis) => {
          axis.style.display = axis.dataset.studioAxisOutput === output ? '' : 'none';
        });
      };
      const bind = () => {
        const video = videoRoot.querySelector('video');
        const svg = root.querySelector('[data-studio-timeline]');
        if (!video || !svg) return;
        if (state.video !== video) {
          state.video?.removeEventListener('timeupdate', state.updateCursor);
          state.video?.removeEventListener('seeking', state.updateCursor);
          state.video = video;
          state.updateCursor = () => {
            const start = Number(svg.dataset.start), end = Number(svg.dataset.end);
            const duration = Number(video.duration);
            const limit = Number.isFinite(duration) && duration > 0 ? duration : end;
            const time = Math.max(0, Math.min(limit, Number(video.currentTime || 0)));
            const viewBox = svg.viewBox.baseVal;
            const progress = Math.max(0, Math.min(1, (time - start) / Math.max(end - start, 1e-9)));
            const x = viewBox.x + progress * viewBox.width;
            const cursor = svg.querySelector('[data-studio-cursor]');
            cursor?.setAttribute('x1', String(x)); cursor?.setAttribute('x2', String(x));
          };
          video.addEventListener('timeupdate', state.updateCursor);
          video.addEventListener('seeking', state.updateCursor);
        }
        if (state.svg === svg && state.clickVideo === video) {
          applyVisibility(); state.updateCursor(); return;
        }
        state.svg?.removeEventListener('click', state.clickSvg);
        state.svg = svg; state.clickVideo = video;
        state.clickSvg = (event) => {
          const point = svg.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
          const matrix = svg.getScreenCTM(); if (!matrix) return;
          const transformed = point.matrixTransform(matrix.inverse());
          const start = Number(svg.dataset.start), end = Number(svg.dataset.end);
          const duration = Number(video.duration);
          const limit = Number.isFinite(duration) && duration > 0 ? duration : end;
          const target = start + (transformed.x - svg.viewBox.baseVal.x) / svg.viewBox.baseVal.width * (end - start);
          video.currentTime = Math.max(0, Math.min(limit, target));
        };
        svg.addEventListener('click', state.clickSvg);
        root.querySelector('[data-studio-output]')?.addEventListener('change', applyVisibility);
        root.querySelectorAll('[data-studio-source]').forEach((box) => box.addEventListener('change', applyVisibility));
        applyVisibility(); state.updateCursor();
      };
      bind();
      if (!state.observer) {
        state.observer = new MutationObserver(bind);
        state.observer.observe(root, {childList: true, subtree: true});
      }
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
            build_timeline_html(result["timeline"], result.get("timeline_duration")),
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
