from pathlib import Path

from chimera_ml.core.config import ExperimentConfig
from chimera_ml.utils.sweep import SweepTarget, resolve_sweep_log_root


def test_resolve_sweep_log_root_uses_console_file_logger_only():
    cfg = ExperimentConfig(
        {
            "logging": [{"name": "console_file_logger", "params": {"log_path": "main_logs"}}],
            "callbacks": [
                {"name": "snapshot_callback", "params": {"log_path": "snapshot_logs"}},
                {"name": "checkpoint_callback", "params": {"log_path": "checkpoint_logs"}},
            ],
        }
    )

    assert resolve_sweep_log_root(cfg) == Path("main_logs")


def test_sweep_target_accepts_objective_alias():
    target = SweepTarget.from_config({"objective": {"monitor": "val/score", "mode": "maximize"}})

    assert target.monitor == "val/score"
    assert target.mode == "max"
