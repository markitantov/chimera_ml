"""Central place to register built-in components.

The library uses small registries (losses, metrics, models, optimizers, ...)
to make components configurable by name.

Registries are populated on import of the corresponding modules.
This helper keeps the "import for side-effects" in one explicit place.
"""


def register_all() -> None:
    """Register all components
    """
    # Losses, metrics, models
    import chimera_ml.losses  
    import chimera_ml.metrics  
    import chimera_ml.models  

    # Training components
    import chimera_ml.training.optimizers  
    import chimera_ml.training.schedulers  

    # Callbacks
    import chimera_ml.callbacks.checkpoint_callback  
    import chimera_ml.callbacks.early_stopping_callback  
    import chimera_ml.callbacks.mlflow_predictions_callback  

    # Data components
    import chimera_ml.data.masking_collate  

    # Loggers
    import chimera_ml.logging.mlflow_logger  
    import chimera_ml.logging.console_file_logger  

    try:
        from chimera_ml.utils.entrypoints import load_entrypoint_plugins
        load_entrypoint_plugins()
    except Exception:
        pass