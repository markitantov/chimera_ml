from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True)
class TrainRunResult:
    """Result returned by the CLI training flow."""

    run_name: str
    target_value: float | None = None
    target_epoch: int | None = None


@dataclass(frozen=True)
class SweepTarget:
    """Log key and direction optimized by an Optuna sweep."""

    monitor: str = "val/loss"
    mode: str = "min"

    def __post_init__(self) -> None:
        monitor = str(self.monitor).strip()
        mode = str(self.mode).strip().lower()
        mode = {"minimize": "min", "maximize": "max"}.get(mode, mode)

        if not monitor:
            raise ValueError("Sweep target monitor cannot be empty.")

        if mode not in ("min", "max"):
            raise ValueError("Sweep target mode must be 'min'/'max' or 'minimize'/'maximize'.")

        object.__setattr__(self, "monitor", monitor)
        object.__setattr__(self, "mode", mode)

    @classmethod
    def from_config(cls, sweep_cfg: Mapping[str, Any]) -> Self:
        raw = sweep_cfg.get("target", sweep_cfg.get("objective", sweep_cfg.get("metric", {}))) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("Sweep config target/objective/metric must be a mapping when provided.")

        return cls(
            monitor=str(raw.get("monitor", sweep_cfg.get("monitor", "val/loss"))),
            mode=str(raw.get("mode", sweep_cfg.get("mode", "min"))),
        )


def resolve_sweep_log_root(cfg: Any) -> Path:
    logger_cfg = cfg.section("logging", name="console_file_logger")
    params = logger_cfg.get("params", {}) if logger_cfg else {}
    if isinstance(params, Mapping) and params.get("log_path"):
        return Path(params["log_path"])

    return Path("logs")


def format_sweep_overrides(overrides: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in overrides.items())


def required_sweep_spec_value(spec: Mapping[str, Any], path: str, key: str) -> Any:
    if key not in spec:
        raise ValueError(f"Optuna parameter '{path}' must define '{key}'.")

    return spec[key]
