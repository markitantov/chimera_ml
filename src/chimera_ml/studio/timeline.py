"""Prediction and XAI timeline rendering."""

import json
from importlib import resources
from typing import Any


def render_template(template_name: str, context: dict[str, Any]) -> str:
    """Render a packaged Studio template with escaped dynamic values."""
    from jinja2 import DictLoader, Environment, select_autoescape

    resource = resources.files("chimera_ml.studio").joinpath("assets", "templates", template_name)
    source = resource.read_text(encoding="utf-8")
    environment = Environment(loader=DictLoader({template_name: source}), autoescape=select_autoescape(("html", "xml")))
    return environment.get_template(template_name).render(**context)


def load_timeline_js() -> str:
    """Load browser synchronization code from package resources."""
    return resources.files("chimera_ml.studio").joinpath("assets", "js", "timeline.js").read_text(encoding="utf-8")


def load_studio_css() -> str:
    """Load Studio component styles from package resources."""
    return resources.files("chimera_ml.studio").joinpath("assets", "css", "studio.css").read_text(encoding="utf-8")


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
    series = []
    for source, source_series in sorted(timeline.items()):
        for output, points in sorted(source_series.items()):
            path = []
            for point in points:
                if point["value"] is None:
                    continue
                x1 = left + (point["start"] - start) / span * plot_width
                x2 = left + (point["end"] - start) / span * plot_width
                y = height - bottom - (point["value"] - low) / value_span * plot_height
                path.append(f"M{x1:.3f},{y:.3f} L{x2:.3f},{y:.3f}")
            series.append(
                {
                    "source": source,
                    "output": output,
                    "points_json": json.dumps(points, separators=(",", ":")),
                    "path": " ".join(path),
                    "color": source_colors[source],
                }
            )
    ticks = []
    for tick in range(6):
        value = low + value_span * tick / 5
        y = height - bottom - (value - low) / value_span * plot_height
        x = left + plot_width * tick / 5
        ticks.append({"x": x, "y": y, "time": start + span * tick / 5, "value": value})
    zero_y = height - bottom - (-low / value_span) * plot_height
    return render_template(
        "xai_timeline.html",
        {
            "width": width,
            "height": height,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "plot_width": plot_width,
            "start": start,
            "maximum": maximum,
            "outputs": outputs,
            "series": series,
            "ticks": ticks,
            "zero_y": zero_y,
        },
    )


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
    axes = []
    for index, output in enumerate(output_names):
        low, high = output_ranges[output]
        y_ticks = []
        for tick in range(6):
            value = low + (high - low) * tick / 5
            y_ticks.append({"value": value, "y": height - bottom - plot_height * tick / 5})
        axes.append({"output": output, "display": "" if index == 0 else "none", "y_ticks": y_ticks})
    series = []
    for source, source_series in sorted(timeline.items()):
        for output, points in sorted(source_series.items()):
            low, high = output_ranges[output]
            path_parts = []
            point_parts = []
            for point in points:
                x1 = left + (point["start"] - minimum) / span * plot_width
                x2 = left + (point["end"] - minimum) / span * plot_width
                y = height - bottom - (point["value"] - low) / max(high - low, 1e-9) * plot_height
                if point["start"] == point["end"]:
                    point_parts.append({"x": x1, "y": y})
                else:
                    path_parts.append(f"M{x1:.3f},{y:.3f} L{x2:.3f},{y:.3f}")
            series.append(
                {
                    "source": source,
                    "output": output,
                    "points_json": json.dumps(points, separators=(",", ":")),
                    "path": " ".join(path_parts),
                    "point_parts": point_parts,
                    "color": source_colors[source],
                }
            )
    x_ticks = [{"x": left + plot_width * tick / 5, "time": minimum + span * tick / 5} for tick in range(6)]
    legend = [
        {"source": source, "color": color, "x": index * 145}
        for index, (source, color) in enumerate(source_colors.items())
    ]
    return render_template(
        "prediction_timeline.html",
        {
            "width": width,
            "height": height,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "plot_width": plot_width,
            "plot_height": plot_height,
            "minimum": minimum,
            "maximum": maximum,
            "duration": duration_sec or maximum,
            "output_names": output_names,
            "axes": axes,
            "series": series,
            "x_ticks": x_ticks,
            "legend": legend,
        },
    )
