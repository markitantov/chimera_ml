"""Prediction and XAI timeline rendering."""

import html
import json
from importlib import resources
from typing import Any


def load_timeline_js() -> str:
    """Load browser synchronization code from package resources."""
    return resources.files("chimera_ml.studio").joinpath("assets/timeline.js").read_text(encoding="utf-8")


def build_xai_timeline_html(
    timeline: dict[str, dict[str, list[dict[str, Any]]]], start: float = 0.0, end: float | None = None
) -> str:
    """Render signed temporal XAI deltas using explicit sample intervals."""
    if not timeline:
        return '<div class="studio-empty-xai">No temporally linked XAI samples available.</div>'
    all_points = [point for series in timeline.values() for points in series.values() for point in points]
    maximum = end if end is not None else max(point["end"] for point in all_points)
    maximum = max(maximum, start + 1e-6)
    width, height = 960, 360
    left, right, top, bottom = 70, 25, 30, 55
    plot_width, plot_height = width - left - right, height - top - bottom
    outputs = sorted({output for series in timeline.values() for output in series})
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2"]
    source_colors = {source: colors[index % len(colors)] for index, source in enumerate(sorted(timeline))}
    values = [point["value"] for point in all_points if point["value"] is not None]
    if not values:
        return '<div class="studio-empty-xai">XAI samples have no numeric temporal delta.</div>'
    low, high = min(min(values), 0.0), max(max(values), 0.0)
    if low == high:
        low, high = -1.0, 1.0
    span = max(maximum - start, 1e-6)
    value_span = high - low
    parts = ['<div class="studio-xai-timeline-controls"><span>XAI output</span><select data-studio-xai-output>']
    parts.extend(
        f'<option value="{html.escape(output, quote=True)}">{html.escape(output)}</option>' for output in outputs
    )
    parts.append("</select></div>")
    zero_y = height - bottom - (-low / value_span) * plot_height
    parts.append(
        f'<svg data-studio-xai-timeline viewBox="0 0 {width} {height}" data-start="{start}" data-end="{maximum}" '
        f'data-plot-left="{left}" data-plot-width="{plot_width}">'
        f'<line data-studio-xai-zero x1="{left}" x2="{width - right}" y1="{zero_y}" y2="{zero_y}" '
        'stroke="#6b7280" stroke-dasharray="4 3" />'
        f'<line data-studio-xai-cursor x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" '
        'stroke="#111827" stroke-width="2" />'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" stroke="#374151" />'
        f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" stroke="#374151" />'
        f'<text x="12" y="{top - 8}">signed delta</text>'
    )
    for tick in range(6):
        value = low + value_span * tick / 5
        y = height - bottom - (value - low) / value_span * plot_height
        x = left + plot_width * tick / 5
        parts.append(
            f'<line x1="{x:.3f}" x2="{x:.3f}" y1="{height - bottom}" y2="{height - bottom + 4}" stroke="#374151" />'
            f'<text x="{x:.3f}" y="{height - bottom + 18}" text-anchor="middle" font-size="11">'
            f"{start + span * tick / 5:.3g}</text>"
            f'<text x="{left - 8}" y="{y + 4:.3f}" text-anchor="end" font-size="11">{value:.3g}</text>'
        )
    for source, series in sorted(timeline.items()):
        for output, points in sorted(series.items()):
            path = []
            for point in points:
                if point["value"] is None:
                    continue
                x1 = left + (point["start"] - start) / span * plot_width
                x2 = left + (point["end"] - start) / span * plot_width
                y = height - bottom - (point["value"] - low) / value_span * plot_height
                path.append(f"M{x1:.3f},{y:.3f} L{x2:.3f},{y:.3f}")
            data = html.escape(json.dumps(points, separators=(",", ":")), quote=True)
            parts.append(
                f'<path data-studio-xai-series data-source="{html.escape(source, quote=True)}" '
                f'data-output="{html.escape(output, quote=True)}" data-points="{data}" d="{" ".join(path)}" '
                f'fill="none" stroke="{source_colors[source]}" stroke-width="2" />'
            )
    parts.append("</svg>")
    return "".join(parts)


def build_timeline_html(
    timeline: dict[str, dict[str, list[dict[str, float]]]], duration_sec: float | None = None
) -> str:
    """Render interval series and browser-side controls as an SVG timeline."""
    if not timeline:
        return '<div class="studio-empty-timeline">No temporal predictions available.</div>'
    all_points = [point for series in timeline.values() for points in series.values() for point in points]
    minimum = 0.0
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
        f'data-start="{minimum}" data-end="{maximum}" data-video-duration="{duration_sec or maximum}" '
        f'data-plot-left="{left}" data-plot-width="{plot_width}">'
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
        for tick in range(6):
            x = left + plot_width * tick / 5
            time_value = minimum + span * tick / 5
            parts.append(
                f'<line x1="{x:.3f}" x2="{x:.3f}" y1="{height - bottom}" y2="{height - bottom + 4}" stroke="#374151" />'
                f'<text x="{x:.3f}" y="{height - bottom + 18}" text-anchor="middle" font-size="11">'
                f"{time_value:.3g}</text>"
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
