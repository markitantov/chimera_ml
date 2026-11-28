"""Prediction overview and temporal prediction extraction."""

from collections.abc import Iterable
from typing import Any


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
