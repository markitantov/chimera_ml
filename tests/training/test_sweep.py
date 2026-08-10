from types import SimpleNamespace

import yaml

from chimera_ml.core.config import ExperimentConfig
from chimera_ml.training.sweep import GridSweep, OptunaSweep


class _TrialStub:
    number = 1

    def __init__(self):
        self.params = {}

    def suggest_float(self, name, low, high, step=None, log=False):
        value = high if log else low
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high, step=1, log=False):
        value = low + step
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = choices[-1]
        self.params[name] = value
        return value


def _base_config(log_path: str = "logs") -> ExperimentConfig:
    return ExperimentConfig(
        {
            "experiment_info": {"params": {"experiment_name": "exp"}},
            "logging": [{"name": "console_file_logger", "params": {"log_path": log_path}}],
            "optimizer": {"name": "adamw_optimizer", "params": {"lr": 0.001}},
            "train": {"params": {"epochs": 3}},
            "model": {"name": "model", "params": {"hidden_dim": 128}},
        }
    )


def test_grid_sweep_writes_expected_artifact_structure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    base_cfg = _base_config()
    sweep_cfg = ExperimentConfig(
        {
            "parameters": {
                "optimizer.params.lr": [0.001, 0.0001],
                "train.params.epochs": [3],
            }
        }
    )
    sweep_run = GridSweep(
        base_cfg=base_cfg,
        sweep_cfg=sweep_cfg,
        experiment_name="exp",
        sweep_name="lr-search",
        timezone="UTC",
        max_trials=None,
    )

    sweep_run.start()
    trial_id, trial_path, trial_cfg = sweep_run.write_trial_config(1, sweep_run.overrides[0])
    sweep_run.save_trial_record({"trial_id": trial_id, "run_name": f"run-{trial_id}"})
    sweep_run.finish("completed")

    assert sweep_run.sweep_dir.resolve() == tmp_path / "logs" / "exp" / "_sweeps" / sweep_run.sweep_id
    assert (sweep_run.sweep_dir / "base_config.yaml").exists()
    assert (sweep_run.sweep_dir / "sweep_config.yaml").exists()
    assert (sweep_run.sweep_dir / "manifest.yaml").exists()
    assert (sweep_run.sweep_dir / "trial_configs").is_dir()
    assert trial_path == sweep_run.sweep_dir / "trial_configs" / f"{trial_id}.yaml"
    assert trial_cfg.raw["optimizer"]["params"]["lr"] == 0.001

    manifest = yaml.safe_load((sweep_run.sweep_dir / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["runs"] == [{"trial_id": trial_id, "run_name": f"run-{trial_id}"}]


def test_grid_sweep_resolves_log_root_from_first_trial_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    sweep_run = GridSweep(
        base_cfg=_base_config("logs"),
        sweep_cfg=ExperimentConfig({"trials": [{"logging.console_file_logger.params.log_path": "sweep_logs"}]}),
        experiment_name="exp",
        sweep_name="root-check",
        timezone="UTC",
        max_trials=None,
    )

    assert sweep_run.sweep_dir.parent.resolve() == tmp_path / "sweep_logs" / "exp" / "_sweeps"


def test_optuna_sweep_suggests_overrides_and_records_best_trial(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    sweep_cfg = ExperimentConfig(
        {
            "method": "optuna",
            "n_trials": 5,
            "objective": {"monitor": "val/score", "mode": "max"},
            "study_name": "manual-study-name",
            "parameters": {
                "optimizer.params.lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "train.params.epochs": {"type": "int", "low": 3, "high": 9, "step": 2},
                "model.params.hidden_dim": {"type": "categorical", "choices": [128, 256, 512]},
            },
        }
    )
    sweep_run = OptunaSweep(
        base_cfg=_base_config(),
        sweep_cfg=sweep_cfg,
        experiment_name="exp",
        sweep_name="opt",
        timezone="UTC",
        max_trials=2,
    )

    overrides = sweep_run.suggest_overrides(_TrialStub())

    assert sweep_run.n_trials == 2
    assert sweep_run.target.monitor == "val/score"
    assert sweep_run.target.mode == "max"
    assert overrides == {
        "optimizer.params.lr": 0.01,
        "train.params.epochs": 5,
        "model.params.hidden_dim": 512,
    }

    sweep_run.start()
    sweep_run.save_best_trial(
        SimpleNamespace(
            number=1,
            value=0.91,
            params={"optimizer.params.lr": 0.01},
            user_attrs={"trial_id": "opt-abcd-002", "run_name": "run-opt-abcd-002", "target_epoch": 4},
        )
    )
    sweep_run.finish("completed")

    manifest = yaml.safe_load((sweep_run.sweep_dir / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["method"] == "optuna"
    assert manifest["study_name"] == "manual-study-name"
    assert manifest["objective"] == {"monitor": "val/score", "mode": "max"}
    assert manifest["n_trials"] == 2
    assert manifest["best_trial"] == {
        "trial_number": 1,
        "trial_id": "opt-abcd-002",
        "run_name": "run-opt-abcd-002",
        "value": 0.91,
        "params": {"optimizer.params.lr": 0.01},
        "target_epoch": 4,
    }
