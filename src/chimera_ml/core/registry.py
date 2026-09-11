from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Name-to-factory registry used by configuration builders.

    A registry owns unique string keys and returns the registered callable.
    Framework registries such as MODELS, LOSSES, CALLBACKS, and INFERENCE_STEPS
    share this contract.
    """

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Return a decorator that registers a callable under key.

        Raises:
            KeyError: If key is already registered.
        """

        def deco(obj: Callable[..., T]) -> Callable[..., T]:
            if key in self._items:
                raise KeyError(f"{self.name}: key '{key}' already registered.")
            self._items[key] = obj
            return obj

        return deco

    def get(self, key: str) -> Callable[..., Any]:
        """Return the callable registered under key.

        Raises:
            KeyError: If key is unknown; the error lists known keys.
        """
        if key not in self._items:
            known = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"{self.name}: unknown key '{key}'. Known: {known}")
        return self._items[key]

    def create(self, key: str, **kwargs: Any) -> Any:
        """Call the factory registered under key with keyword arguments.

        Args:
            key: Registry key.
            kwargs: Factory arguments.
        Returns:
            Factory result.
        """
        factory = self.get(key)
        return factory(**kwargs)

    def keys(self) -> list[str]:
        """Return registered keys in deterministic sorted order."""
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
