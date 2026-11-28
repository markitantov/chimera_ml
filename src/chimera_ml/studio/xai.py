"""Rendering and extraction of existing XAI artifacts."""

import html
from collections.abc import Iterable
from typing import Any


def iter_xai_values(value: Any, prefix: str = "") -> Iterable[tuple[str, float | None]]:
    """Yield labeled scalar values without changing nulls into numeric values."""
    if value is None:
        yield prefix or "value", None
    elif isinstance(value, dict):
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_xai_values(item, label)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            label = f"{prefix}[{index}]" if prefix else str(index)
            yield from iter_xai_values(item, label)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix or "value", float(value)


def render_xai_overview(xai: Any) -> str:
    """Render existing XAI samples as a non-causal signed-delta overview."""
    if not isinstance(xai, dict) or not xai.get("samples"):
        return '<div class="studio-empty-xai">No XAI artifacts available for this inference.</div>'
    policy = html.escape(str(xai.get("ablation_policy", "unknown")))
    parts = [f'<div class="studio-xai-overview"><p>Ablation policy: <code>{policy}</code></p>']
    for sample in xai["samples"]:
        sample_id = html.escape(str(sample.get("sample_id", "unknown")))
        baseline = ", ".join(
            f"{html.escape(label)}={value if value is not None else 'null'}"
            for label, value in iter_xai_values(sample.get("baseline"))
        )
        parts.append(f'<section class="studio-xai-sample"><h4>Sample {sample_id}</h4><p>Baseline: {baseline}</p>')
        modalities = sample.get("modalities", {})
        numeric_deltas = [
            value
            for result in modalities.values()
            for _, value in iter_xai_values(result.get("delta"))
            if value is not None
        ]
        scale = max((abs(value) for value in numeric_deltas), default=1.0)
        if scale == 0.0:
            scale = 1.0

        for modality, result in modalities.items():
            status = str(result.get("status", "unknown"))
            reason = result.get("reason")
            reason_text = f" — {html.escape(str(reason))}" if reason else ""
            parts.append(
                f"<div><strong>{html.escape(str(modality))}</strong> <em>{html.escape(status)}</em>{reason_text}"
            )
            for label, value in iter_xai_values(result.get("delta")):
                if value is None:
                    parts.append(f"<p>{html.escape(label)}: null</p>")
                    continue
                width = min(abs(value) / scale * 45.0, 45.0)
                side = "positive" if value >= 0 else "negative"
                left = "50%" if value >= 0 else f"calc(50% - {width:.2f}%)"
                parts.append(
                    f'<div class="studio-xai-bar-row"><span>{html.escape(label)}</span>'
                    '<span class="studio-xai-bar-track" style="display:inline-block;position:relative;'
                    'width:180px;height:12px;background:#e5e7eb">'
                    '<span class="studio-xai-zero" style="position:absolute;left:50%;top:0;'
                    'height:100%;border-left:1px solid #111827"></span>'
                    f'<span class="studio-xai-bar {side}" style="position:absolute;top:0;height:100%;'
                    f'left:{left};width:{width:.2f}%"></span></span>'
                    f"<span>{value:+.6g}</span></div>"
                )
            parts.append("</div>")
        parts.append("</section>")
    parts.append("</div>")
    return "".join(parts)


def collect_temporal_records(value: Any, result: dict[str, tuple[float, float]]) -> None:
    """Collect explicit sample-id time ranges from inference artifacts."""
    if isinstance(value, dict):
        sample_id = value.get("sample_id", value.get("id"))
        start = value.get("start_sec", value.get("start", value.get("time_sec", value.get("timestamp"))))
        end = value.get("end_sec", value.get("end", start))
        if sample_id is not None and isinstance(start, (int, float)) and isinstance(end, (int, float)):
            result[str(sample_id)] = (float(start), float(end))
        for item in value.values():
            collect_temporal_records(item, result)
    elif isinstance(value, list):
        for item in value:
            collect_temporal_records(item, result)


def extract_xai_timeline(xai: Any, artifacts: Any, predictions: Any) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Extract temporal XAI only when samples have explicit ID-based time ranges."""
    if not isinstance(xai, dict) or not isinstance(xai.get("samples"), list):
        return {}
    temporal: dict[str, tuple[float, float]] = {}
    for source in (artifacts, predictions):
        if isinstance(source, dict):
            for key, value in source.items():
                if key != "xai":
                    collect_temporal_records(value, temporal)
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for sample in xai["samples"]:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id", ""))
        bounds = temporal.get(sample_id)
        if bounds is None:
            start = sample.get("start_sec", sample.get("start", sample.get("time_sec")))
            end = sample.get("end_sec", sample.get("end", start))
            bounds = (
                (float(start), float(end))
                if isinstance(start, (int, float)) and isinstance(end, (int, float))
                else None
            )
        if bounds is None:
            continue
        for modality, explanation in sample.get("modalities", {}).items():
            if not isinstance(explanation, dict) or explanation.get("status") != "evaluated":
                continue
            for output, value in iter_xai_values(explanation.get("delta")):
                result.setdefault(str(modality), {}).setdefault(output, []).append(
                    {"start": bounds[0], "end": bounds[1], "value": value, "sample_id": sample_id}
                )
    return result
