from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

from chimera_ml.core.config import ExperimentConfig
from chimera_ml.logging.utils import local_datetime_tag, short_hash
from chimera_ml.utils.sweep import SweepTarget, required_sweep_spec_value, resolve_sweep_log_root


class _SweepBase:
    def __init__(
        self,
        *,
        base_cfg: ExperimentConfig,
        sweep_cfg: ExperimentConfig,
        experiment_name: str,
        sweep_name: str | None,
        timezone: str | None,
        log_root_cfg: ExperimentConfig | None = None,
    ) -> None:
        self.base_cfg = base_cfg
        self.sweep_cfg = sweep_cfg
        self.experiment_name = experiment_name
        self.timezone = timezone
        self.label = (sweep_name or "sweep").strip() or "sweep"

        date_tag = local_datetime_tag(fmt="%y%m%d-%H%M", timezone=timezone)
        started_at = local_datetime_tag(fmt="%Y-%m-%d_%H-%M-%S", timezone=timezone)
        hash_time = local_datetime_tag(fmt="%Y-%m-%d_%H-%M-%S-%f", timezone=timezone)
        short_id = short_hash(f"{sweep_cfg.to_yaml_text()}\n{hash_time}", n=4)

        self.short_id = short_id
        self.sweep_id = f"{self.label}-{date_tag}-{short_id}"
        self.started_at = started_at
        self.sweep_dir = resolve_sweep_log_root(log_root_cfg or base_cfg) / experiment_name / "_sweeps" / self.sweep_id
        self.manifest_path = self.sweep_dir / "manifest.yaml"
        self.manifest = {
            "sweep_id": self.sweep_id,
            "sweep_name": self.label,
            "base_config": "base_config.yaml",
            "sweep_config": "sweep_config.yaml",
            "started_at": started_at,
            "finished_at": None,
            "status": "running",
            "runs": [],
        }

    def start(self) -> None:
        self.sweep_dir.mkdir(parents=True, exist_ok=False)
        self.base_cfg.to_yaml(self.sweep_dir / "base_config.yaml")
        self.sweep_cfg.to_yaml(self.sweep_dir / "sweep_config.yaml")
        (self.sweep_dir / "trial_configs").mkdir()
        ExperimentConfig(self.manifest).to_yaml(self.manifest_path)

    def finish(self, status: str) -> None:
        self.manifest["finished_at"] = local_datetime_tag(fmt="%Y-%m-%d_%H-%M-%S", timezone=self.timezone)
        self.manifest["status"] = status
        ExperimentConfig(self.manifest).to_yaml(self.manifest_path)

    def write_trial_config(self, index: int, overrides: Mapping[str, Any]) -> tuple[str, Path, ExperimentConfig]:
        trial_id = f"{self.label}-{self.short_id}-{index:03d}"
        trial_cfg = self.base_cfg.copy()
        trial_cfg.apply_overrides(overrides)
        trial_path = self.sweep_dir / "trial_configs" / f"{trial_id}.yaml"
        trial_cfg.to_yaml(trial_path)
        return trial_id, trial_path, trial_cfg

    def save_trial_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(record)
        self.manifest["runs"].append(record)
        ExperimentConfig(self.manifest).to_yaml(self.manifest_path)
        return record


class GridSweep(_SweepBase):
    def __init__(
        self,
        *,
        base_cfg: ExperimentConfig,
        sweep_cfg: ExperimentConfig,
        experiment_name: str,
        sweep_name: str | None,
        timezone: str | None,
        max_trials: int | None,
    ) -> None:
        overrides = self._overrides_from_config(sweep_cfg.raw)
        if max_trials is not None:
            overrides = overrides[:max_trials]

        if not overrides:
            raise ValueError("Sweep config produced no trials.")

        first_trial_cfg = base_cfg.copy()
        first_trial_cfg.apply_overrides(overrides[0])
        super().__init__(
            base_cfg=base_cfg,
            sweep_cfg=sweep_cfg,
            experiment_name=experiment_name,
            sweep_name=sweep_name,
            timezone=timezone,
            log_root_cfg=first_trial_cfg,
        )
        self.overrides = overrides

    @classmethod
    def _overrides_from_config(cls, sweep_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
        trials = sweep_cfg.get("trials")
        parameters = sweep_cfg.get("parameters")

        if trials is not None and parameters is not None:
            raise ValueError("Sweep config must define either 'trials' or 'parameters', not both.")

        if trials is not None:
            if not isinstance(trials, Sequence) or isinstance(trials, (str, bytes)):
                raise TypeError("Sweep config 'trials' must be a list of mappings.")

            for i, trial in enumerate(trials, start=1):
                if not isinstance(trial, Mapping):
                    raise TypeError(f"Sweep trial #{i} must be a mapping.")

            return [dict(trial) for trial in trials]

        if not isinstance(parameters, Mapping):
            raise TypeError("Sweep config must contain a 'parameters' mapping or a 'trials' list.")

        paths = list(parameters.keys())
        value_lists: list[list[Any]] = []
        for path in paths:
            values = parameters[path]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise TypeError(f"Sweep parameter '{path}' must be a non-empty list.")

            if not values:
                raise ValueError(f"Sweep parameter '{path}' must contain at least one value.")

            value_lists.append(list(values))

        return [dict(zip(paths, values, strict=True)) for values in product(*value_lists)]


class OptunaSweep(_SweepBase):
    DEFAULT_TRIALS = 10

    def __init__(
        self,
        *,
        base_cfg: ExperimentConfig,
        sweep_cfg: ExperimentConfig,
        experiment_name: str,
        sweep_name: str | None,
        timezone: str | None,
        max_trials: int | None,
    ) -> None:
        raw = sweep_cfg.raw
        if raw.get("trials") is not None:
            raise ValueError("Optuna sweep configs must use 'parameters', not explicit 'trials'.")

        search_space = raw.get("parameters")
        if not isinstance(search_space, Mapping) or not search_space:
            raise TypeError("Optuna sweep config must contain a non-empty 'parameters' mapping.")

        super().__init__(
            base_cfg=base_cfg,
            sweep_cfg=sweep_cfg,
            experiment_name=experiment_name,
            sweep_name=sweep_name,
            timezone=timezone,
        )
        self.search_space = search_space
        self.target = SweepTarget.from_config(raw)
        self.n_trials = self._n_trials(raw, max_trials)
        self.study_name = str(raw.get("study_name") or self.sweep_id)
        self.storage = raw.get("storage")
        self.load_if_exists = bool(raw.get("load_if_exists", False))
        self.manifest.update(
            {
                "method": "optuna",
                "study_name": self.study_name,
                "objective": {"monitor": self.target.monitor, "mode": self.target.mode},
                "n_trials": self.n_trials,
            }
        )

    def create_study(self) -> Any:
        optuna = self._import_optuna()
        direction = "minimize" if self.target.mode == "min" else "maximize"
        return optuna.create_study(
            direction=direction,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
        )

    def suggest_overrides(self, trial: Any) -> dict[str, Any]:
        suggestions: dict[str, Any] = {}
        for path, spec in self.search_space.items():
            suggestions[str(path)] = self._suggest_value(trial, str(path), spec)

        return suggestions

    def save_best_trial(self, best_trial: Any) -> None:
        attrs = getattr(best_trial, "user_attrs", {}) or {}
        self.manifest["best_trial"] = {
            "trial_number": int(best_trial.number),
            "trial_id": attrs.get("trial_id"),
            "run_name": attrs.get("run_name"),
            "value": float(best_trial.value),
            "params": dict(getattr(best_trial, "params", {})),
        }

        if attrs.get("target_epoch") is not None:
            self.manifest["best_trial"]["target_epoch"] = attrs["target_epoch"]

    def _n_trials(self, sweep_cfg: Mapping[str, Any], max_trials: int | None) -> int:
        configured = sweep_cfg.get("n_trials", max_trials if max_trials is not None else self.DEFAULT_TRIALS)
        n_trials = int(configured)
        if max_trials is not None:
            n_trials = min(n_trials, max_trials)

        if n_trials < 1:
            raise ValueError("Optuna sweep must run at least one trial.")

        return n_trials

    def _import_optuna(self) -> Any:
        try:
            import optuna  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Dependency 'optuna' is not installed. Install it with: pip install optuna"
            ) from exc

        return optuna

    def _suggest_value(self, trial: Any, path: str, spec: Any) -> Any:
        if not isinstance(spec, Mapping):
            raise TypeError(f"Optuna parameter '{path}' must be a mapping.")

        kind = str(spec.get("type", "")).strip().lower()
        if not kind:
            kind = "categorical" if "choices" in spec else "float"

        if kind in ("float", "suggest_float"):
            low = float(required_sweep_spec_value(spec, path, "low"))
            high = float(required_sweep_spec_value(spec, path, "high"))
            step = spec.get("step")
            log = bool(spec.get("log", False))
            if step is not None and log:
                raise ValueError(f"Optuna parameter '{path}' cannot use both 'step' and 'log'.")

            return trial.suggest_float(path, low, high, step=None if step is None else float(step), log=log)

        if kind in ("int", "integer", "suggest_int"):
            low = int(required_sweep_spec_value(spec, path, "low"))
            high = int(required_sweep_spec_value(spec, path, "high"))
            step = int(spec.get("step", 1))
            log = bool(spec.get("log", False))
            if step < 1:
                raise ValueError(f"Optuna parameter '{path}' step must be >= 1.")

            if log and step != 1:
                raise ValueError(f"Optuna parameter '{path}' cannot use 'log: true' with step != 1.")

            return trial.suggest_int(path, low, high, step=step, log=log)

        if kind in ("categorical", "choice", "choices", "suggest_categorical"):
            choices = required_sweep_spec_value(spec, path, "choices")
            if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
                raise ValueError(f"Optuna parameter '{path}' choices must be a non-empty list.")

            return trial.suggest_categorical(path, list(choices))

        raise ValueError(f"Unsupported Optuna parameter type '{kind}' for '{path}'.")
