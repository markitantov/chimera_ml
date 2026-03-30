from abc import ABC, abstractmethod
from typing import Any


class BaseLogger(ABC):
    """
    Abstract base class for loggers used by Trainer.
    """

    @abstractmethod
    def start(self, params: dict[str, Any] | None = None) -> None:
        """Start logging session (safe to call multiple times)."""
        return

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics at given step."""
        return

    @abstractmethod
    def end(self) -> None:
        """Finish logging session (safe to call multiple times)."""
        return

    # Optional capability: artifacts
    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        """Log a file as an artifact (default: no-op)."""
        return

    # Optional capability: text
    def log_text(self, text: str, artifact_path: str, filename: str) -> None:
        """Log text as artifact (default: no-op)."""
        return

    # Optional capability: bytes
    def log_artifact_bytes(self, data: bytes, artifact_path: str, filename: str) -> None:
        """Log bytes as artifact (default: no-op)."""
        return
