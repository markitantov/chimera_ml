from chimera_ml.studio.predictions import build_prediction_overview, extract_timeline
from chimera_ml.studio.timeline import build_timeline_html


def test_prediction_overview_is_task_agnostic() -> None:
    overview = build_prediction_overview(
        {
            "emotion": {"label": "happy", "class_index": 1, "probabilities": {"sad": 0.1, "happy": 0.9}},
            "age": {"value": 32.5},
            "text": "hello",
            "segments": [{"start_sec": 0.0}],
        }
    )

    assert overview["emotion"]["label"] == "happy"
    assert overview["emotion"]["probabilities"]["happy"] == 0.9
    assert overview["age"]["value"] == 32.5
    assert "segments" not in overview


def test_timeline_keeps_real_intervals_and_gaps() -> None:
    artifacts = {
        "modality_predictions": {
            "audio": [
                {"start_sec": 0.0, "end_sec": 1.0, "score": 0.2},
                {"start_sec": 2.0, "end_sec": 3.5, "score": 0.8},
            ],
            "video": [{"start_sec": 0.0, "end_sec": 1.5, "score": 0.4}],
        }
    }
    predictions = {"segments": [{"start_sec": 0.5, "end_sec": 1.25, "result": {"probabilities": {"x": 0.7}}}]}

    timeline = extract_timeline(artifacts, predictions)
    assert len(timeline["audio"]["score"]) == 2
    assert timeline["audio"]["score"][1]["start"] == 2.0
    assert timeline["fusion"]["result.x"][0]["value"] == 0.7
    markup = build_timeline_html(timeline)
    assert 'data-start="0.0"' in markup
    assert 'data-source="audio"' in markup
    assert 'data-source="video"' in markup
    assert 'data-output="score"' in markup


def test_timeline_without_temporal_predictions_is_explicit() -> None:
    assert extract_timeline({}, {"label": "happy"}) == {}
    assert "No temporal predictions" in build_timeline_html({})


def test_timeline_excludes_service_numeric_fields() -> None:
    timeline = extract_timeline(
        {},
        {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "duration_sec": 1.0,
                    "num_frames": 20,
                    "has_faces": True,
                    "emotion": {"probabilities": {"happy": 0.7}},
                }
            ]
        },
    )
    assert "emotion.happy" in timeline["fusion"]
    assert all("duration" not in key and "num_" not in key for key in timeline["fusion"])
