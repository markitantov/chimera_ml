from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file into a Python dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ExperimentConfig:
    """Convenient wrapper around raw YAML dictionary."""

    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """Load an experiment config from a YAML file."""
        raw = load_yaml(path)
        if not isinstance(raw, dict):
            raise TypeError("Experiment config must be a YAML mapping.")

        return cls(raw)

    def copy(self) -> Self:
        """Return a deep copy of this config wrapper."""
        return type(self)(deepcopy(self.raw))

    def to_yaml_text(self) -> str:
        """Serialize this config to YAML text."""
        return yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=True)

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file."""
        Path(path).write_text(self.to_yaml_text(), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level config value with optional default."""
        return self.raw.get(key, default)

    def section(self, key: str, *, name: str | None = None) -> dict[str, Any]:
        """Return a config section handling dict and list-with-name formats."""
        value = self.raw.get(key, {})
        if value is None:
            return {}

        if isinstance(value, dict):
            return value

        if isinstance(value, list):
            if name is None:
                raise TypeError(
                    f"Config section '{key}' is a list; pass name='...' "
                    f"(e.g., cfg.section('logging', name='mlflow_logger'))."
                )

            for item in value:
                if isinstance(item, dict) and item.get("name") == name:
                    return item

            return {}

        raise TypeError(f"Config section '{key}' must be a dict or a list, got {type(value)}.")

    def validate(self, *, require_experiment_name: bool = True) -> list[str]:
        """Return human-readable config validation errors."""
        errors: list[str] = []
        raw = self.raw

        if not isinstance(raw, dict):
            return ["Top-level config must be a mapping/object."]

        # Named sections required by current train/eval builders.
        for section in ("data", "model", "train", "loss", "optimizer"):
            value = raw.get(section)
            if not isinstance(value, dict):
                errors.append(f"Section '{section}' must be a mapping.")
                continue

            if section != "train" and not value.get("name"):
                errors.append(f"Section '{section}' must contain non-empty 'name'.")

            params = value.get("params", {})
            if params is None:
                params = {}

            if not isinstance(params, dict):
                errors.append(f"Section '{section}.params' must be a mapping.")

        scheduler = raw.get("scheduler")
        if scheduler is not None:
            if not isinstance(scheduler, dict):
                errors.append("Section 'scheduler' must be a mapping when provided.")
            elif not scheduler.get("name"):
                errors.append("Section 'scheduler' must contain non-empty 'name' when provided.")

        for section in ("metrics", "callbacks", "logging"):
            value = raw.get(section)
            if value is None:
                continue

            if not isinstance(value, list):
                errors.append(f"Section '{section}' must be a list.")
                continue

            for i, item in enumerate(value):
                if not isinstance(item, dict):
                    errors.append(f"Section '{section}[{i}]' must be a mapping.")
                    continue

                if not item.get("name"):
                    errors.append(f"Section '{section}[{i}]' must contain non-empty 'name'.")

        if require_experiment_name:
            experiment_info = raw.get("experiment_info")
            if not isinstance(experiment_info, dict):
                errors.append("Section 'experiment_info' must be a mapping.")
            else:
                params = experiment_info.get("params", {})
                if not isinstance(params, dict):
                    errors.append("Section 'experiment_info.params' must be a mapping.")

                elif not params.get("experiment_name"):
                    errors.append("`experiment_info.params.experiment_name` is required.")

        return errors

    def set_at_path(self, path: str, value: Any) -> None:
        """Set a config value using dotted paths and named list entries."""
        parts = [part.strip() for part in path.split(".") if part.strip()]
        if not parts:
            raise ValueError("Config path cannot be empty.")

        node: Any = self.raw
        for part in parts[:-1]:
            if isinstance(node, dict):
                if part not in node or node[part] is None:
                    node[part] = {}

                node = node[part]
                continue

            if isinstance(node, list):
                if part.isdigit():
                    index = int(part)
                    try:
                        node = node[index]
                    except IndexError as exc:
                        raise ValueError(f"List index {index} is out of range in path '{path}'.") from exc
                    continue

                match = next((item for item in node if isinstance(item, dict) and item.get("name") == part), None)
                if match is None:
                    match = {"name": part}
                    node.append(match)

                node = match
                continue

            raise ValueError(f"Cannot descend into '{part}' in path '{path}': got {type(node).__name__}.")

        leaf = parts[-1]
        if isinstance(node, dict):
            node[leaf] = value
            return

        if isinstance(node, list):
            if leaf.isdigit():
                index = int(leaf)
                try:
                    node[index] = value
                except IndexError as exc:
                    raise ValueError(f"List index {index} is out of range in path '{path}'.") from exc
                return

            for i, item in enumerate(node):
                if isinstance(item, dict) and item.get("name") == leaf:
                    node[i] = value
                    return

            node.append({"name": leaf, "params": value if isinstance(value, dict) else {"value": value}})
            return

        raise ValueError(f"Cannot set '{leaf}' in path '{path}': got {type(node).__name__}.")

    def apply_overrides(self, overrides: Mapping[str, Any]) -> None:
        """Apply dotted-path config overrides in iteration order."""
        for path, value in overrides.items():
            self.set_at_path(str(path), value)


@dataclass
class TrainConfig:
    """Runtime training configuration used by `Trainer`."""

    epochs: int = 10
    grad_clip_norm: float | None = None
    mixed_precision: bool = False
    log_every_steps: int = 50
    device: str = "cuda"  # "cuda"|"cpu"

    # Multiple train loaders
    train_loader_mode: str = "single"  # How to sample when train_loaders has multiple loaders:
    # single|round_robin|weighted
    train_stop_on: str = "min"  # When to end an epoch in multi-loader mode:
    # min=stop on first exhausted, max=stop on last exhausted
    train_loader_weights: dict[str, float] | None = None  # Per-loader sampling weights for weighted mode:
    # {loader_name: weight}

    # Scheduler
    use_scheduler: bool = False
    scheduler_step_per_epoch: bool = True
    scheduler_monitor: str | None = None

    # Predictions caching (for callbacks / multiple val hooks without recomputation)
    collect_cache: bool = True
