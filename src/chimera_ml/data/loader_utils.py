"""Utilities for normalizing dataloader inputs into a stable mapping format."""

import re
from collections.abc import Mapping, Sequence

from torch.utils.data import DataLoader


def sanitize_split_name(name: str) -> str:
    """Normalize split names for stable metric/artifact keys.

    The function keeps alphanumerics, dot, underscore, and dash;
    all other characters are replaced with `_`.
    """
    normalized = name.strip()
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    return normalized or "split"


def normalize_loaders(
        loaders: DataLoader | Mapping[str, DataLoader] | Sequence[DataLoader] | None, 
        *, 
        default_name: str
    ) -> dict[str, DataLoader]:
    """Normalize dataloader containers into a `name -> loader` dictionary."""
    if loaders is None:
        return {}

    if isinstance(loaders, DataLoader):
        return {default_name: loaders}

    if isinstance(loaders, Mapping):
        out: dict[str, DataLoader] = {}
        for key_raw, loader in loaders.items():
            key = sanitize_split_name(str(key_raw))
            if key in out:
                i = 2
                new_key = f"{key}_{i}"
                while new_key in out:
                    i += 1
                    new_key = f"{key}_{i}"

                key = new_key

            out[key] = loader

        return out

    if isinstance(loaders, (list, tuple)):
        return {f"{default_name}{i}": loader for i, loader in enumerate(loaders)}

    raise TypeError(f"Unsupported loaders type: {type(loaders)}")
