"""Utilities for scheduling batches from multiple train loaders."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch


def iter_mixed_train_batches(
    loaders: Mapping[str, Iterable[Any]],
    *,
    mode: str,
    stop_on: str,
    train_loader_weights: Mapping[str, float] | None = None,
) -> Iterator[tuple[str, Any]]:
    """Yield `(loader_name, batch)` according to the selected multi-loader mode.

    Modes:
    - `single`: consume only the first loader.
    - `round_robin`: cycle through loaders in order.
    - `weighted`: sample next loader by weights.
    """
    names = list(loaders.keys())
    if not names:
        return

    mode = (mode or "single").lower()
    stop_on = (stop_on or "min").lower()
    if stop_on not in {"min", "max"}:
        raise ValueError("train_stop_on must be 'min' or 'max'")

    if mode == "single":
        name = names[0]
        for b in loaders[name]:
            yield name, b
        return

    if mode == "round_robin":
        iters = {n: iter(loaders[n]) for n in names}
        active = set(names)
        while active:
            for n in names:
                if n not in active:
                    continue
                try:
                    yield n, next(iters[n])
                except StopIteration:
                    if stop_on == "min":
                        return
                    active.remove(n)
        return

    if mode == "weighted":
        weights_cfg = train_loader_weights or {}
        iters = {n: iter(loaders[n]) for n in names}
        active = set(names)
        min_steps_budget: int | None = None
        if stop_on == "min":
            lengths = [_safe_len(loaders[n]) for n in names]
            if all(v is not None for v in lengths):
                min_steps_budget = min(int(v) for v in lengths if v is not None)
        yielded = 0
        while active:
            if min_steps_budget is not None and yielded >= min_steps_budget:
                return

            active_names = [n for n in names if n in active]
            ws = torch.tensor([float(weights_cfg.get(n, 1.0)) for n in active_names], dtype=torch.float32)
            ws = torch.clamp(ws, min=0.0)
            if float(ws.sum().item()) <= 0:
                ws = torch.ones_like(ws)
            idx = int(torch.multinomial(ws / ws.sum(), num_samples=1).item())
            chosen = active_names[idx]
            try:
                yield chosen, next(iters[chosen])
                yielded += 1
            except StopIteration:
                if stop_on == "min":
                    return
                active.remove(chosen)
        return

    raise NotImplementedError(f"train_loader_mode='{mode}' is not supported. Use one of: single|round_robin|weighted.")


def _safe_len(x: Any) -> int | None:
    """Return `len(x)` as `int` when available, otherwise `None`."""
    try:
        return len(x)
    except Exception:
        return None


def estimate_train_epoch_steps(
    loaders: Mapping[str, Iterable[Any]],
    *,
    mode: str,
    stop_on: str,
) -> int | None:
    """Estimate expected number of train steps for a mixed-loader epoch."""
    lengths = {n: _safe_len(dl) for n, dl in loaders.items()}
    if any(v is None for v in lengths.values()):
        return None

    vals = [v for v in lengths.values() if v is not None]
    if not vals:
        return None

    mode = (mode or "single").lower()
    stop_on = (stop_on or "min").lower()

    if mode == "single":
        first_name = next(iter(loaders.keys()))
        return lengths[first_name]

    if mode == "round_robin":
        if stop_on == "min":
            return min(vals) * len(loaders)
        return sum(vals)

    if mode == "weighted":
        if stop_on == "min":
            return min(vals)
        return sum(vals)

    return None
