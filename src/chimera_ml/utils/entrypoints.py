import warnings

_LOADED: set[str] = set()


def load_entrypoint_plugins(group: str = "chimera_ml.plugins") -> None:
    """Load external plugins declared through Python entry points.

    Args:
        group: Entry-point group to select, defaulting to
            chimera_ml.plugins.
    Notes:
        Each entry point may expose a callable registration function or a
        module-level object. Callable objects are invoked once; failures emit
        warnings instead of aborting the whole startup.
    """
    try:
        from importlib.metadata import entry_points  # py3.10+
    except Exception:  # pragma: no cover
        try:
            from importlib_metadata import entry_points  # type: ignore
        except Exception:
            return

    try:
        eps = entry_points()
        if not hasattr(eps, "select"):
            return
        selected = list(eps.select(group=group))
    except Exception:
        return

    for ep in selected:
        ep_id = f"{group}:{getattr(ep, 'name', 'unknown')}"
        if ep_id in _LOADED:
            continue
        _LOADED.add(ep_id)

        try:
            obj = ep.load()
            if callable(obj):
                obj()
        except Exception as e:
            warnings.warn(
                f"Failed to load entry point plugin '{ep_id}': {e}",
                stacklevel=2,
            )
