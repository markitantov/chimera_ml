from chimera_ml.data.loader_utils import normalize_loaders
from chimera_ml.training.mixed_loader_utils import (
    estimate_train_epoch_steps,
    iter_mixed_train_batches,
)


def test_iter_mixed_train_batches_round_robin_min():
    loaders = {"a": [1, 2], "b": [10, 20, 30]}

    got = list(iter_mixed_train_batches(loaders, mode="round_robin", stop_on="min"))
    assert [name for name, _ in got] == ["a", "b", "a", "b"]


def test_iter_mixed_train_batches_round_robin_max():
    loaders = {"a": [1, 2], "b": [10, 20, 30]}

    got = list(iter_mixed_train_batches(loaders, mode="round_robin", stop_on="max"))
    assert [name for name, _ in got] == ["a", "b", "a", "b", "b"]


def test_iter_mixed_train_batches_single():
    loaders = {"a": [1, 2], "b": [10, 20, 30]}

    got = list(iter_mixed_train_batches(loaders, mode="single", stop_on="min"))
    assert [name for name, _ in got] == ["a", "a"]


def test_estimate_train_epoch_steps():
    loaders = {"a": [1, 2], "b": [10, 20, 30]}

    assert estimate_train_epoch_steps(loaders, mode="single", stop_on="min") == 2
    assert estimate_train_epoch_steps(loaders, mode="round_robin", stop_on="min") == 4
    assert estimate_train_epoch_steps(loaders, mode="round_robin", stop_on="max") == 5
    assert estimate_train_epoch_steps(loaders, mode="weighted", stop_on="min") == 2
    assert estimate_train_epoch_steps(loaders, mode="weighted", stop_on="max") == 5


def test_normalize_loaders_for_list_input():
    out = normalize_loaders([[1, 2], [3]], default_name="val")
    assert list(out.keys()) == ["val0", "val1"]
