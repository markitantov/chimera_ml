import sys
import types

import pytest

from chimera_ml import plugins


def test_builtin_modules_include_all_callback_and_collate_registrations():
    assert "chimera_ml.callbacks.snapshot_callback" in plugins._BUILTIN_MODULES
    assert "chimera_ml.data.masking_collate" in plugins._BUILTIN_MODULES


def test_register_all_calls_entrypoint_loader(monkeypatch):
    called = {"n": 0}

    fake_module = types.SimpleNamespace(load_entrypoint_plugins=lambda: called.__setitem__("n", called["n"] + 1))
    monkeypatch.setitem(sys.modules, "chimera_ml.utils.entrypoints", fake_module)

    plugins.register_all()

    assert called["n"] == 1


def test_register_all_swallows_loader_errors(monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    fake_module = types.SimpleNamespace(load_entrypoint_plugins=_boom)
    monkeypatch.setitem(sys.modules, "chimera_ml.utils.entrypoints", fake_module)

    with pytest.warns(UserWarning, match="Failed to load entrypoint plugins"):
        plugins.register_all()
