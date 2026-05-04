from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InferenceContext:
    """Shared state passed through inference steps."""

    input_path: Path
    work_dir: Path
    device: str
    config: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)

    def get_artifact(self, name: str, default: Any = None) -> Any:
        return self.artifacts.get(name, default)

    def set_artifact(self, name: str, value: Any) -> None:
        self.artifacts[name] = value

    @property
    def predictions(self) -> Any | None:
        return self.get_artifact("predictions")

    @predictions.setter
    def predictions(self, value: Any) -> None:
        self.set_artifact("predictions", value)
