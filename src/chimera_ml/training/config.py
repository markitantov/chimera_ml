from dataclasses import dataclass


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
    train_loader_weights: dict[str, float] | None = None  # Per-loader sampling weights for weighted mode:
    # {loader_name: weight}

    # Scheduler
    use_scheduler: bool = False
    scheduler_step_per_epoch: bool = True
    scheduler_monitor: str | None = None

    # Predictions caching (for callbacks / multiple val hooks without recomputation)
    collect_cache: bool = True
