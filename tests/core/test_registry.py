import pytest

from chimera_ml.core.registry import Registry


def test_registry_register_get_create_keys():
    reg = Registry("demo")

    @reg.register("x")
    def build_x(v: int = 1):
        return {"v": v}

    assert reg.get("x") is build_x
    assert reg.create("x", v=5) == {"v": 5}
    assert reg.keys() == ["x"]


def test_registry_duplicate_key_raises():
    reg = Registry("demo")

    @reg.register("x")
    def _a():
        return 1

    with pytest.raises(KeyError):

        @reg.register("x")
        def _b():
            return 2


def test_registry_unknown_key_raises_with_known_list():
    reg = Registry("demo")
    with pytest.raises(KeyError) as e:
        reg.get("missing")
    assert "unknown key" in str(e.value)
