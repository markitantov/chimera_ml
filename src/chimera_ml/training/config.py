from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Runtime configuration consumed by Trainer.

    Attributes:
        epochs: Number of one-based training epochs.
        grad_clip_norm: Optional maximum gradient norm; None disables clipping.
        mixed_precision: Enable autocast and gradient scaling on CUDA.
        log_every_steps: Intended interval for progress logging.
        device: Requested device string, typically cuda or cpu.
        train_loader_mode: Multi-loader strategy: single, round_robin, or
            weighted.
        train_stop_on: Whether a multi-loader epoch ends at first (min) or
            last (max) exhausted loader.
        train_loader_weights: Optional per-loader weights for weighted mode.
        use_scheduler: Whether Trainer steps the configured scheduler.
        scheduler_step_per_epoch: Step once per epoch instead of after each
            optimizer update.
        scheduler_monitor: Optional metric for metric-aware schedulers.
        collect_cache: Cache split predictions and metadata for callbacks.
    """

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
