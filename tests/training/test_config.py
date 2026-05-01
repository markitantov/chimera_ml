from pathlib import Path

import pytest

from chimera_ml.core.config import ExperimentConfig, load_yaml
from chimera_ml.training import ExperimentConfig as TrainingExperimentConfig


def test_load_yaml_reads_file(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text("seed: 42\nmodel:\n  name: demo\n", encoding="utf-8")
    data = load_yaml(str(p))
    assert data["seed"] == 42
    assert data["model"]["name"] == "demo"


def test_experiment_config_from_yaml_reads_mapping(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text("seed: 42\nmodel:\n  name: demo\n", encoding="utf-8")
    cfg = ExperimentConfig.from_yaml(str(p))

    assert cfg.get("seed") == 42
    assert cfg.section("model")["name"] == "demo"


def test_experiment_config_serializes_yaml(tmp_path: Path):
    cfg = ExperimentConfig({"seed": 42, "experiment_info": {"params": {"experiment_name": "тест"}}})

    text = cfg.to_yaml_text()
    out = tmp_path / "out.yaml"
    cfg.to_yaml(out)
    loaded = ExperimentConfig.from_yaml(str(out))

    assert text.startswith("seed: 42")
    assert "experiment_name: тест" in text
    assert loaded.get("seed") == 42
    assert loaded.section("experiment_info")["params"]["experiment_name"] == "тест"


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


def test_experiment_config_validate_success():
    cfg = ExperimentConfig(
        {
            "experiment_info": {"params": {"experiment_name": "exp"}},
            "data": {"name": "dm", "params": {}},
            "model": {"name": "m", "params": {}},
            "train": {"params": {"epochs": 1}},
            "loss": {"name": "loss", "params": {}},
            "optimizer": {"name": "opt", "params": {}},
            "metrics": [{"name": "metric", "params": {}}],
        }
    )

    assert cfg.validate(require_experiment_name=True) == []


def test_experiment_config_validate_reports_missing_experiment_name():
    cfg = ExperimentConfig(
        {
            "experiment_info": {"params": {}},
            "data": {"name": "dm", "params": {}},
            "model": {"name": "m", "params": {}},
            "train": {"params": {}},
            "loss": {"name": "loss", "params": {}},
            "optimizer": {"name": "opt", "params": {}},
        }
    )

    assert "`experiment_info.params.experiment_name` is required." in cfg.validate(require_experiment_name=True)


def test_set_at_path_updates_nested_mapping_values():
    raw = {"optimizer": {"params": {"lr": 0.001}}}
    cfg = ExperimentConfig(raw)

    cfg.set_at_path("optimizer.params.lr", 0.0001)

    assert raw["optimizer"]["params"]["lr"] == 0.0001


def test_set_at_path_updates_named_list_entries():
    raw = {
        "callbacks": [
            {"name": "checkpoint_callback", "params": {"monitor": "val/loss"}},
            {"name": "early_stopping_callback", "params": {"patience": 2}},
        ]
    }
    cfg = ExperimentConfig(raw)

    cfg.set_at_path("callbacks.checkpoint_callback.params.run_name", "r1")

    assert raw["callbacks"][0]["params"]["run_name"] == "r1"
    assert "run_name" not in raw["callbacks"][1]["params"]


def test_apply_overrides_updates_named_list_entries():
    raw = {
        "callbacks": [
            {"name": "checkpoint_callback", "params": {"monitor": "val/loss"}},
            {"name": "early_stopping_callback", "params": {"patience": 2}},
        ]
    }
    cfg = ExperimentConfig(raw)

    cfg.apply_overrides({"callbacks.checkpoint_callback.params.monitor": "val/ccc"})

    assert raw["callbacks"][0]["params"]["monitor"] == "val/ccc"
    assert raw["callbacks"][1]["params"]["patience"] == 2


def test_training_import_experiment_config_is_backward_compatible():
    assert TrainingExperimentConfig is ExperimentConfig
