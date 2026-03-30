import torch
from torch.utils.data import Dataset

from chimera_ml.data.datamodule import DataModule


class _ToyDataset(Dataset):
    def __init__(self, n: int = 4):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        return {
            "inputs": {"x": torch.tensor([float(idx)])},
            "targets": torch.tensor([idx], dtype=torch.float32),
        }


def test_datamodule_train_val_test_return_none_when_missing():
    dm = DataModule(train_dataset=None, val_dataset=None, test_dataset=None)

    assert dm.train_dataloader() is None
    assert dm.val_dataloader() is None
    assert dm.test_dataloader() is None


def test_datamodule_builds_single_loader_with_train_flags():
    ds = _ToyDataset(5)
    dm = DataModule(
        train_dataset=ds,
        batch_size=2,
        shuffle_train=True,
        drop_last_train=True,
        num_workers=0,
        pin_memory=False,
    )

    loader = dm.train_dataloader()

    assert loader.batch_size == 2
    assert loader.drop_last is True


def test_datamodule_supports_mapping_and_sequence_datasets():
    train_map = {"a": _ToyDataset(3), "b": _ToyDataset(2)}
    val_seq = [_ToyDataset(2), _ToyDataset(1)]
    test_ds = _ToyDataset(4)

    dm = DataModule(
        train_dataset=train_map,
        val_dataset=val_seq,
        test_dataset=test_ds,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    train_loaders = dm.train_dataloader()
    val_loaders = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    assert isinstance(train_loaders, dict)
    assert set(train_loaders.keys()) == {"a", "b"}

    assert isinstance(val_loaders, list)
    assert len(val_loaders) == 2

    assert test_loader.batch_size == 2
