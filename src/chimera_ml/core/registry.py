from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Simple name -> factory registry for plug-and-play components."""

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to register a factory/class under a key."""

        def deco(obj: Callable[..., T]) -> Callable[..., T]:
            if key in self._items:
                raise KeyError(f"{self.name}: key '{key}' already registered.")
            self._items[key] = obj
            return obj

        return deco

    def get(self, key: str) -> Callable[..., Any]:
        """Return a registered factory by key."""
        if key not in self._items:
            known = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"{self.name}: unknown key '{key}'. Known: {known}")
        return self._items[key]

    def create(self, key: str, **kwargs: Any) -> Any:
        """Create an object from a registered factory."""
        factory = self.get(key)
        return factory(**kwargs)

    def keys(self) -> list[str]:
        """Return sorted registry keys."""
        return sorted(self._items.keys())


# Training component registries
MODELS = Registry("models")
LOSSES = Registry("losses")
METRICS = Registry("metrics")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
CALLBACKS = Registry("callbacks")

# Data & logging registries
DATAMODULES = Registry("datamodules")
COLLATES = Registry("collates")
LOGGERS = Registry("loggers")
INFERENCE_STEPS = Registry("inference_steps")
