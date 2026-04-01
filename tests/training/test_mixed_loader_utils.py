import pytest
import torch

from chimera_ml.training.mixed_loader_utils import (
    estimate_train_epoch_steps,
    iter_mixed_train_batches,
)


def test_iter_mixed_train_batches_returns_empty_for_no_loaders():
    got = list(iter_mixed_train_batches({}, mode="single", stop_on="min"))
    assert got == []


def test_iter_mixed_train_batches_invalid_stop_on_raises():
    loaders = {"a": [1, 2]}
    with pytest.raises(ValueError, match="train_stop_on"):
        list(iter_mixed_train_batches(loaders, mode="single", stop_on="bad"))


def test_iter_mixed_train_batches_unknown_mode_raises():
    loaders = {"a": [1, 2]}
    with pytest.raises(NotImplementedError, match="train_loader_mode"):
        list(iter_mixed_train_batches(loaders, mode="unknown", stop_on="min"))


def test_iter_mixed_train_batches_weighted_min_stops_on_first_exhausted(monkeypatch):
    loaders = {"a": [1], "b": [10, 20, 30]}

    # Always pick the first active loader ("a"), so stop_on=min should end early.
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.tensor([0]))
    got = list(
        iter_mixed_train_batches(
            loaders,
            mode="weighted",
            stop_on="min",
            train_loader_weights={"a": 1.0, "b": 1.0},
        )
    )
    assert [name for name, _ in got] == ["a"]


def test_iter_mixed_train_batches_weighted_max_continues_after_one_loader_exhausted(monkeypatch):
    loaders = {"a": [1], "b": [10, 20]}

    # Always pick the first active loader.
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.tensor([0]))
    got = list(
        iter_mixed_train_batches(
            loaders,
            mode="weighted",
            stop_on="max",
            train_loader_weights={"a": 1.0, "b": 1.0},
        )
    )
    assert [name for name, _ in got] == ["a", "b", "b"]


def test_iter_mixed_train_batches_weighted_with_non_positive_weights_falls_back_to_uniform(monkeypatch):
    loaders = {"a": [1], "b": [10]}

    # If implementation falls back to uniform, chosen index 1 should select loader "b".
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.tensor([1]))
    got = list(
        iter_mixed_train_batches(
            loaders,
            mode="weighted",
            stop_on="min",
            train_loader_weights={"a": 0.0, "b": -5.0},
        )
    )
    assert [name for name, _ in got] == ["b"]


def test_iter_mixed_train_batches_weighted_min_does_not_exceed_shortest_epoch_budget(monkeypatch):
    loaders = {"a": [1], "b": [10, 20, 30]}

    # Keep sampling the longer loader; stop_on=min should still end after min loader budget.
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.tensor([1]))
    got = list(
        iter_mixed_train_batches(
            loaders,
            mode="weighted",
            stop_on="min",
            train_loader_weights={"a": 1.0, "b": 1.0},
        )
    )
    assert len(got) == 1
    assert [name for name, _ in got] == ["b"]


class _NoLenIterable:
    def __iter__(self):
        yield 1
        yield 2


def test_estimate_train_epoch_steps_returns_none_when_len_missing():
    loaders = {"a": [1, 2], "b": _NoLenIterable()}
    assert estimate_train_epoch_steps(loaders, mode="round_robin", stop_on="min") is None


def test_estimate_train_epoch_steps_returns_none_for_empty_mapping():
    assert estimate_train_epoch_steps({}, mode="single", stop_on="min") is None
