from dataclasses import dataclass
from typing import Any, Dict, Optional, Iterable, Union, Sequence

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config into a python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ExperimentConfig:
    """Convenient wrapper around raw YAML dict."""
    raw: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def section(self, key: str, *, name: Optional[str] = None) -> Dict[str, Any]:
        """
        - If raw[key] is a dict -> return it (old behavior).
        - If raw[key] is a list of {"name": ..., "params": ...}:
            - if name is provided -> return the first matching item (full dict)
            - else -> error (to avoid ambiguity)
        """
        v = self.raw.get(key, {})
        if v is None:
            return {}

        # old behavior
        if isinstance(v, dict):
            return v

        # new behavior for list sections (e.g., logging)
        if isinstance(v, list):
            if name is None:
                raise TypeError(
                    f"Config section '{key}' is a list; pass name='...' "
                    f"(e.g., cfg.section('logging', name='mlflow_logger'))."
                )

            for item in v:
                if isinstance(item, dict) and item.get("name") == name:
                    return item
            return {}

        raise TypeError(f"Config section '{key}' must be a dict or a list, got {type(v)}.")
    
    def patch_params_at(self, path: str, names: list[str], **kwargs) -> None:
        node: Any = self.raw
        for k in path.split("."):
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                return

        if not isinstance(node, list):
            return

        for item in node:
            if isinstance(item, dict) and item.get("name") in names:
                item.setdefault("params", {}).update(kwargs)
