import pytest

from chimera_ml.callbacks.sweep_target_callback import SweepTargetCallback


def test_sweep_target_callback_tracks_best_min_value():
    cb = SweepTargetCallback(monitor="val/loss", mode="min")

    cb.on_fit_start(object())
    cb.on_epoch_end(object(), 1, {"val/loss": 1.0})
    cb.on_epoch_end(object(), 2, {"val/loss": 0.75})
    cb.on_epoch_end(object(), 3, {"val/loss": 0.9})

    assert cb.best_value == 0.75
    assert cb.best_epoch == 2
    assert cb.last_value == 0.9
    assert cb.last_epoch == 3


def test_sweep_target_callback_tracks_best_max_value():
    cb = SweepTargetCallback(monitor="val/score", mode="max")

    cb.on_fit_start(object())
    cb.on_epoch_end(object(), 1, {"val/score": 0.2})
    cb.on_epoch_end(object(), 2, {"val/score": 0.8})

    assert cb.best_value == 0.8
    assert cb.best_epoch == 2


def test_sweep_target_callback_keeps_available_keys_when_monitor_missing():
    cb = SweepTargetCallback(monitor="val/loss", mode="min")

    cb.on_fit_start(object())
    cb.on_epoch_end(object(), 1, {"train/loss": 1.0})

    assert cb.best_value is None
    assert cb.available_keys == ("train/loss",)


def test_sweep_target_callback_rejects_non_scalar_monitor():
    cb = SweepTargetCallback(monitor="val/loss", mode="min")

    with pytest.raises(ValueError, match="scalar-convertible"):
        cb.on_epoch_end(object(), 1, {"val/loss": object()})
