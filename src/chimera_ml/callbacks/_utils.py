from collections.abc import Sequence
from typing import Any

from torch.utils.data import DataLoader


def resolve_splits(
    trainer: Any,
    splits: Sequence[str] | None,
) -> list[tuple[str, DataLoader | None]]:
    """Resolve split selectors into concrete `(split_name, loader)` pairs."""
    out: list[tuple[str, DataLoader | None]] = []
    seen: set[str] = set()

    def add(name: str, loader: DataLoader | None) -> None:
        if name and name not in seen:
            seen.add(name)
            out.append((name, loader))

    train_loaders = getattr(trainer, "_train_loaders", None)
    val_loaders = getattr(trainer, "_val_loaders", None)
    test_loaders = getattr(trainer, "_test_loaders", None)
    all_loaders = getattr(trainer, "_loaders", None)

    selectors: list[str] = (
        [splits] if isinstance(splits, str) else list(splits) if splits else ["val"]
    )

    for split in selectors:
        if split == "train":
            if isinstance(train_loaders, dict) and train_loaders:
                for name, loader in train_loaders.items():
                    add(name, loader)
            elif isinstance(all_loaders, dict):
                add("train", all_loaders.get("train"))
            continue

        if split == "val":
            if isinstance(val_loaders, dict) and val_loaders:
                for name, loader in val_loaders.items():
                    add(name, loader)
            elif isinstance(all_loaders, dict):
                add("val", all_loaders.get("val"))
            continue

        if split == "test":
            if isinstance(test_loaders, dict) and test_loaders:
                for name, loader in test_loaders.items():
                    add(name, loader)
            elif isinstance(all_loaders, dict):
                add("test", all_loaders.get("test"))
            continue

        loader = None
        if isinstance(train_loaders, dict) and split in train_loaders:
            loader = train_loaders[split]
        elif isinstance(val_loaders, dict) and split in val_loaders:
            loader = val_loaders[split]
        elif isinstance(test_loaders, dict) and split in test_loaders:
            loader = test_loaders[split]
        elif isinstance(all_loaders, dict) and split in all_loaders:
            loader = all_loaders[split]

        add(split, loader)

    return out
