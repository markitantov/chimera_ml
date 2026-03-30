from dataclasses import dataclass
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file into a Python dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ExperimentConfig:
    """Convenient wrapper around raw YAML dictionary."""

    raw: dict[str, Any]

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

    def patch_params_at(self, path: str, names: list[str], **kwargs: Any) -> None:
        """Patch `params` for selected named nodes at the given dotted path."""
        node: Any = self.raw
        for key in path.split("."):
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                return

        if not isinstance(node, list):
            return

        for item in node:
            if isinstance(item, dict) and item.get("name") in names:
                item.setdefault("params", {}).update(kwargs)


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
    train_loader_weights: dict[str, float] | None = (
        None  # Per-loader sampling weights for weighted mode:
    )
    # {loader_name: weight}

    # Scheduler
    use_scheduler: bool = False
    scheduler_step_per_epoch: bool = True
    scheduler_monitor: str | None = None

    # Predictions caching (for callbacks / multiple val hooks without recomputation)
    collect_cache: bool = True
