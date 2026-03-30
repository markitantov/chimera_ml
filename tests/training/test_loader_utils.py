import torch

from chimera_ml.data.loader_utils import normalize_loaders
from chimera_ml.training.mixed_loader_utils import (
    estimate_train_epoch_steps,
    iter_mixed_train_batches,
)


def test_normalize_loaders_dict_name_sanitization():
    loaders = {"val full": [1, 2]}
    out = normalize_loaders(loaders, default_name="val")
    assert "val_full" in out


def test_iter_mixed_train_batches_weighted_runs():
    torch.manual_seed(0)
    loaders = {"a": [1, 2], "b": [10, 20, 30]}
    got = list(
        iter_mixed_train_batches(
            loaders,
            mode="weighted",
            stop_on="min",
            train_loader_weights={"a": 0.1, "b": 0.9},
        )
    )
    assert len(got) >= 2
    assert all(name in {"a", "b"} for name, _ in got)


def test_estimate_train_epoch_steps_unknown_mode_returns_none():
    loaders = {"a": [1, 2], "b": [3]}
    assert estimate_train_epoch_steps(loaders, mode="unknown", stop_on="min") is None
