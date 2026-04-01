import importlib.metadata

import pytest

from chimera_ml.utils import entrypoints as ep


class _EntryPointStub:
    def __init__(self, name, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader


class _EntryPointsSelectStub:
    def __init__(self, items):
        self._items = items

    def select(self, group):
        return self._items


def test_load_entrypoint_plugins_calls_callable_once(monkeypatch):
    ep._LOADED.clear()
    called = {"n": 0}

    def _register():
        called["n"] += 1

    eps = _EntryPointsSelectStub([_EntryPointStub("plugin_a", _register)])
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: eps)

    ep.load_entrypoint_plugins("chimera_ml.plugins")
    ep.load_entrypoint_plugins("chimera_ml.plugins")

    assert called["n"] == 1


def test_load_entrypoint_plugins_warns_on_loader_failure(monkeypatch):
    ep._LOADED.clear()

    def _broken_loader():
        raise RuntimeError("boom")

    eps = _EntryPointsSelectStub([_EntryPointStub("plugin_c", _broken_loader)])
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: eps)

    with pytest.warns(UserWarning, match="Failed to load entry point plugin"):
        ep.load_entrypoint_plugins("chimera_ml.plugins")


def test_load_entrypoint_plugins_handles_entry_points_exception(monkeypatch):
    ep._LOADED.clear()

    def _raise():
        raise RuntimeError("entry_points unavailable")

    monkeypatch.setattr(importlib.metadata, "entry_points", _raise)
    ep.load_entrypoint_plugins("chimera_ml.plugins")
