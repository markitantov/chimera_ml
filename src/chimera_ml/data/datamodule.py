from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import DataLoader, Dataset

from chimera_ml.core.batch import Batch
from chimera_ml.data.masking_collate import MaskingCollate


@dataclass
class DataModule:
    """Lightning-like helper that builds train/val/test dataloaders."""

    train_dataset: Dataset | Mapping[str, Dataset] | Sequence[Dataset] | None = None
    val_dataset: Dataset | Mapping[str, Dataset] | Sequence[Dataset] | None = None
    test_dataset: Dataset | Mapping[str, Dataset] | Sequence[Dataset] | None = None

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    shuffle_train: bool = True
    drop_last_train: bool = False

    collate_fn: Callable[[list[dict[str, Any]]], Batch] = field(default_factory=MaskingCollate)

    def _make_loader(self, dataset: Dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        """Create a single DataLoader instance."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
        )

    def _make_loaders(
        self,
        datasets: Dataset | Mapping[str, Dataset] | Sequence[Dataset],
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader | Mapping[str, DataLoader] | Sequence[DataLoader]:
        """Build loaders preserving the input container shape."""
        # Mapping[str, Dataset] -> Mapping[str, DataLoader]
        if isinstance(datasets, Mapping):
            return {name: self._make_loader(ds, shuffle=shuffle, drop_last=drop_last) for name, ds in datasets.items()}

        # Sequence[Dataset] (list/tuple) -> list[DataLoader]
        if isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)):
            # Protect against someone accidentally passing a single Dataset that implements __len__/__getitem__.
            # Most torch Datasets are *not* Sequences, so this is typically safe.
            return [self._make_loader(ds, shuffle=shuffle, drop_last=drop_last) for ds in datasets]

        # Single Dataset
        return self._make_loader(datasets, shuffle=shuffle, drop_last=drop_last)

    def train_dataloader(
        self,
    ) -> DataLoader | Mapping[str, DataLoader] | Sequence[DataLoader] | None:
        """Return train dataloaders or `None` when train dataset is absent."""
        if self.train_dataset is None:
            return None

        return self._make_loaders(self.train_dataset, shuffle=self.shuffle_train, drop_last=self.drop_last_train)

    def val_dataloader(self) -> DataLoader | Mapping[str, DataLoader] | Sequence[DataLoader] | None:
        """Return validation dataloaders or `None` when val dataset is absent."""
        if self.val_dataset is None:
            return None

        return self._make_loaders(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(
        self,
    ) -> DataLoader | Mapping[str, DataLoader] | Sequence[DataLoader] | None:
        """Return test dataloaders or `None` when test dataset is absent."""
        if self.test_dataset is None:
            return None

        return self._make_loaders(self.test_dataset, shuffle=False, drop_last=False)
