from pathlib import Path

import pytest

from chimera_ml.training.config import ExperimentConfig, load_yaml


def test_load_yaml_reads_file(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text("seed: 42\nmodel:\n  name: demo\n", encoding="utf-8")
    data = load_yaml(str(p))
    assert data["seed"] == 42
    assert data["model"]["name"] == "demo"


def test_experiment_config_section_for_dict_and_list():
    cfg = ExperimentConfig(
        {
            "model": {"name": "m"},
            "logging": [
                {"name": "mlflow_logger", "params": {"tracking_uri": "x"}},
                {"name": "console_file_logger", "params": {"log_path": "logs"}},
            ],
        }
    )

    assert cfg.section("model") == {"name": "m"}
    assert cfg.section("logging", name="mlflow_logger")["name"] == "mlflow_logger"
    assert cfg.section("logging", name="missing") == {}


def test_experiment_config_section_list_without_name_raises():
    cfg = ExperimentConfig({"logging": [{"name": "mlflow_logger"}]})
    with pytest.raises(TypeError):
        cfg.section("logging")


def test_patch_params_at_updates_only_selected_items():
    raw = {
        "callbacks": [
            {"name": "checkpoint_callback", "params": {"a": 1}},
            {"name": "early_stopping_callback", "params": {"b": 2}},
        ]
    }
    cfg = ExperimentConfig(raw)
    cfg.patch_params_at("callbacks", names=["checkpoint_callback"], run_name="r1")

    assert raw["callbacks"][0]["params"]["run_name"] == "r1"
    assert "run_name" not in raw["callbacks"][1]["params"]
