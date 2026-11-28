"""Inference execution and result serialization for Studio."""

import tempfile
from pathlib import Path
from typing import Any

from chimera_ml.inference import (
    InferenceConfig,
    InferenceContext,
    build_inference_pipeline,
    resolve_inference_device,
)

from .predictions import build_prediction_overview, extract_timeline
from .xai import extract_xai_timeline


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
        "xai": artifacts.get("xai") if isinstance(artifacts, dict) else None,
        "xai_timeline": extract_xai_timeline(artifacts.get("xai"), artifacts, predictions),
        "timeline_duration": duration_sec,
        "work_dir": str(work_dir),
    }
