import json

import pytest

from chimera_ml.studio.app import create_demo, timeline_sync_js
from chimera_ml.studio.timeline import build_timeline_html


def test_create_demo_registers_browser_sync() -> None:
    pytest.importorskip("gradio")
    demo = create_demo(config_paths=["example.yaml"])
    config = demo.get_config_file()
    serialized = json.dumps(config)
    assert "studio-timeline" in serialized
    assert "currentTime" in serialized
    assert "currentTime" in timeline_sync_js()
    assert "timeupdate" in timeline_sync_js()
    assert "studio-xai-timeline" in timeline_sync_js()
    assert "plotLeft" in timeline_sync_js()


def test_timeline_uses_common_output_scale_and_video_duration() -> None:
    timeline = {
        "audio": {"probability": [{"start": 0.0, "end": 1.0, "value": 0.2}]},
        "video": {"probability": [{"start": 2.0, "end": 2.0, "value": 0.8}]},
    }
    markup = build_timeline_html(timeline, duration_sec=5.0)
    assert 'data-start="0.0"' in markup
    assert 'data-end="5.0"' in markup
    assert 'data-plot-left="70"' in markup
    assert 'data-plot-width="865"' in markup
    assert ">0</text>" in markup
    assert ">1</text>" in markup
    assert 'stroke="#2563eb"' in markup
    assert 'stroke="#dc2626"' in markup
    assert "data-studio-cursor" in markup
    assert "cx=" in markup
    assert "time (s)" in markup


def test_timeline_sync_tears_down_by_reusing_single_observer() -> None:
    script = timeline_sync_js()
    assert "root.__chimeraStudioSync" in script
    assert "if (!state.observer)" in script
    assert "data-studio-source]:checked" in script
    assert "Math.min(limit, target)" in script
    assert "data-studio-axis-output" in script
