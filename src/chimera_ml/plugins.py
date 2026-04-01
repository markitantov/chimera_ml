"""Central place to register built-in components.

The library uses small registries (losses, metrics, models, optimizers, ...)
to make components configurable by name.

Registries are populated on import of the corresponding modules.
This helper keeps the "import for side-effects" in one explicit place.
"""

import importlib
import warnings

_BUILTIN_MODULES: tuple[str, ...] = (
    "chimera_ml.losses",
    "chimera_ml.training.optimizers",
    "chimera_ml.training.schedulers",
    "chimera_ml.models",
    "chimera_ml.metrics",
    "chimera_ml.logging.mlflow_logger",
    "chimera_ml.logging.console_file_logger",
    "chimera_ml.callbacks.checkpoint_callback",
    "chimera_ml.callbacks.collect_predictions_callback",
    "chimera_ml.callbacks.early_stopping_callback",
    "chimera_ml.callbacks.plot_confusion_matrix_callback",
    "chimera_ml.callbacks.snapshot_callback",
    "chimera_ml.callbacks.telegram_notifier_callback",
    "chimera_ml.data.masking_collate",
)

def register_all() -> None:
    """Register all built-in components and entrypoint plugins."""
    for mod in _BUILTIN_MODULES:
        importlib.import_module(mod)

    try:
        from chimera_ml.utils.entrypoints import load_entrypoint_plugins

        load_entrypoint_plugins()
    except Exception as e:
        warnings.warn(f"Failed to load entrypoint plugins: {e}", stacklevel=2)
