import warnings

_LOADED: set[str] = set()


def load_entrypoint_plugins(group: str = "chimera_ml.plugins") -> None:
    """Load external plugins declared via Python entry points.

    Any entry point in the given *group* may point to:
      - a callable (function) that performs registrations, or
      - a module-level object; if it's callable we call it, otherwise we just load it.

    This enables "project plugins" without listing them in YAML/CLI.
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
        if hasattr(eps, "select"):
            selected = list(eps.select(group=group))
        else:  # pragma: no cover (older API)
            selected = list(eps.get(group, []))  # type: ignore[attr-defined]
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
