from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InferenceContext:
    """Mutable state and artifact store shared by inference steps.

    Attributes:
        input_path: Input file or sample path for the current run.
        work_dir: Directory for caches and intermediate artifacts.
        device: Resolved runtime device string.
        config: Raw inference configuration mapping.
        artifacts: Named values produced by steps; predictions is conventional.
        written_artifact_keys: Keys written by the current step context and
            used by the pipeline when merging results.

    Steps should use get_artifact and set_artifact rather than mutating the
    artifact ownership bookkeeping directly.
    """

    input_path: Path
    work_dir: Path
    device: str
    config: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    _written_artifact_keys: set[str] = field(default_factory=set, init=False, repr=False)

    def get_artifact(self, name: str, default: Any = None) -> Any:
        """Return a named artifact or a default when it is absent.

        Args:
            name: Artifact key.
            default: Value returned when the key is missing.
        """
        return self.artifacts.get(name, default)

    def set_artifact(self, name: str, value: Any) -> None:
        """Store an artifact and mark it as written by this step.

        Args:
            name: Artifact key.
            value: Value to store.
        """
        self.artifacts[name] = value
        self._written_artifact_keys.add(name)

    @property
    def written_artifact_keys(self) -> frozenset[str]:
        return frozenset(self._written_artifact_keys)

    @property
    def predictions(self) -> Any | None:
        return self.get_artifact("predictions")

    @predictions.setter
    def predictions(self, value: Any) -> None:
        self.set_artifact("predictions", value)
