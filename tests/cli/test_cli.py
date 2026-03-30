import pytest
import torch
import typer

from chimera_ml import cli


class _DMStub:
    def train_dataloader(self):
        return "train_loader"

    def val_dataloader(self):
        return {"val": "val_loader"}

    def test_dataloader(self):
        return "test_loader"


class _DMEvalStub:
    def train_dataloader(self):
        return {"train": "train_loader"}

    def val_dataloader(self):
        return {"val": "val_loader"}

    def test_dataloader(self):
        return {"test": "test_loader"}


class _DMNestedStub:
    def train_dataloader(self):
        return {"main": "train_loader_main"}

    def val_dataloader(self):
        return ["val_loader_0", "val_loader_1"]

    def test_dataloader(self):
        return {"test_a": "test_loader_a"}


class _TrainCfg:
    def __init__(self):
        self.epochs = 5


class _ModelStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))
        self.loaded = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded = (state_dict, strict)
        return super().load_state_dict({}, strict=False)


class _TrainerStub:
    last_init = None
    last_fit = None
    last_eval = None

    def __init__(self, **kwargs):
        _TrainerStub.last_init = kwargs
        self.model = kwargs["model"]

    def fit(self, train_loader, val_loaders=None):
        _TrainerStub.last_fit = (train_loader, val_loaders)

    def evaluate(self, loaders, with_features=False, feature_extractor=None):
        _TrainerStub.last_eval = (loaders, with_features, feature_extractor)
        return {"ok": 1.0}


def _config_for_train():
    return {
        "seed": 123,
        "experiment_info": {
            "params": {
                "experiment_name": "exp",
                "include_time": False,
                "datetime_format": "%Y-%m-%d",
                "timezone": "UTC",
            }
        },
        "data": {"name": "dm", "params": {}},
        "model": {"name": "m", "params": {}},
        "train": {"name": "train_cfg", "params": {}},
        "loss": {"name": "loss", "params": {}},
        "optimizer": {"name": "opt", "params": {}},
        "scheduler": {"name": "sch", "params": {}},
        "metrics": [{"name": "metric", "params": {}}],
        "callbacks": [
            {"name": "checkpoint_callback", "params": {}},
            {"name": "snapshot_callback", "params": {"save_config": True}},
        ],
        "logging": [
            {"name": "mlflow_logger", "params": {}},
            {"name": "console_file_logger", "params": {}},
        ],
    }


def _config_for_train_without_snapshot():
    cfg = _config_for_train()
    cfg["callbacks"] = [
        {"name": "checkpoint_callback", "params": {}},
    ]
    return cfg


def _config_for_eval():
    return {
        "seed": 321,
        "data": {"name": "dm", "params": {}},
        "model": {"name": "m", "params": {}},
        "train": {"name": "train_cfg", "params": {}},
        "loss": {"name": "loss", "params": {}},
        "optimizer": {"name": "opt", "params": {}},
        "metrics": [{"name": "metric", "params": {}}],
        "callbacks": [{"name": "checkpoint_callback", "params": {}}],
    }


def test_cli_train_wires_builders_and_trainer(monkeypatch):
    model = _ModelStub()

    monkeypatch.setattr(cli, "load_yaml", lambda _: _config_for_train())
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "generate_run_name", lambda **_: "run_name")
    monkeypatch.setattr(cli, "build_datamodule", lambda _: _DMStub())
    monkeypatch.setattr(cli, "build_model", lambda _: model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", lambda _: "loss")
    monkeypatch.setattr(cli, "build_metrics", lambda _: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *_: "opt")
    monkeypatch.setattr(cli, "build_scheduler", lambda *_: "sch")
    monkeypatch.setattr(cli, "build_callbacks", lambda *_: ["cb"])
    monkeypatch.setattr(
        cli, "build_logger", lambda cfg, inject=None: {"cfg": cfg, "inject": inject}
    )
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)

    cli.train(config_path="cfg.yaml", class_names="cat, dog")

    assert _TrainerStub.last_init is not None
    assert _TrainerStub.last_init["class_names"] == ["cat", "dog"]
    assert _TrainerStub.last_init["scheduler"] == "sch"
    assert _TrainerStub.last_fit == ("train_loader", {"val": "val_loader"})


def test_cli_train_works_without_snapshot_callback(monkeypatch):
    model = _ModelStub()

    monkeypatch.setattr(cli, "load_yaml", lambda _: _config_for_train_without_snapshot())
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "generate_run_name", lambda **_: "run_name")
    monkeypatch.setattr(cli, "build_datamodule", lambda _: _DMStub())
    monkeypatch.setattr(cli, "build_model", lambda _: model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", lambda _: "loss")
    monkeypatch.setattr(cli, "build_metrics", lambda _: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *_: "opt")
    monkeypatch.setattr(cli, "build_scheduler", lambda *_: "sch")
    monkeypatch.setattr(cli, "build_callbacks", lambda *_: ["cb"])
    monkeypatch.setattr(
        cli, "build_logger", lambda cfg, inject=None: {"cfg": cfg, "inject": inject}
    )
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)

    cli.train(config_path="cfg.yaml", class_names=None)

    assert _TrainerStub.last_init is not None
    assert _TrainerStub.last_fit == ("train_loader", {"val": "val_loader"})


def test_cli_eval_loads_checkpoint_and_calls_evaluate(monkeypatch):
    model = _ModelStub()

    monkeypatch.setattr(cli, "load_yaml", lambda _: _config_for_eval())
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "build_datamodule", lambda _: _DMEvalStub())
    monkeypatch.setattr(cli, "build_model", lambda _: model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", lambda _: "loss")
    monkeypatch.setattr(cli, "build_metrics", lambda _: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *_: "opt")
    monkeypatch.setattr(cli, "build_callbacks", lambda *_: ["cb"])
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)
    monkeypatch.setattr(cli.torch, "load", lambda *args, **kwargs: {"model_state_dict": {}})

    cli.eval(
        config_path="cfg_eval.yaml",
        checkpoint_path="ckpt.pt",
        with_features=True,
        class_names="c1,c2",
    )

    assert _TrainerStub.last_init is not None
    assert _TrainerStub.last_init["class_names"] == ["c1", "c2"]
    assert _TrainerStub.last_init["mlflow_logger"] is None
    assert _TrainerStub.last_init["logger"] is None
    assert _TrainerStub.last_eval is not None
    loaders, with_features, feature_extractor = _TrainerStub.last_eval
    assert set(loaders.keys()) == {"train", "val", "test"}
    assert with_features is True
    assert feature_extractor is None
    assert model.loaded is not None
    _, strict_flag = model.loaded
    assert strict_flag is True


def test_cli_eval_flattens_nested_loader_containers(monkeypatch):
    model = _ModelStub()

    monkeypatch.setattr(cli, "load_yaml", lambda _: _config_for_eval())
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "build_datamodule", lambda _: _DMNestedStub())
    monkeypatch.setattr(cli, "build_model", lambda _: model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", lambda _: "loss")
    monkeypatch.setattr(cli, "build_metrics", lambda _: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *_: "opt")
    monkeypatch.setattr(cli, "build_callbacks", lambda *_: ["cb"])
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)

    cli.eval(
        config_path="cfg_eval.yaml",
        checkpoint_path=None,
        with_features=False,
        class_names=None,
    )

    assert _TrainerStub.last_eval is not None
    loaders, with_features, feature_extractor = _TrainerStub.last_eval
    assert set(loaders.keys()) == {"train_main", "val0", "val1", "test_a"}
    assert with_features is False
    assert feature_extractor is None


class _RegistryStub:
    def __init__(self, items):
        self._items = list(items)

    def keys(self):
        return sorted(self._items)


class _EntryPointStub:
    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value


def _valid_cli_cfg():
    return {
        "seed": 0,
        "experiment_info": {"params": {"experiment_name": "exp"}},
        "data": {"name": "dm", "params": {}},
        "model": {"name": "m", "params": {}},
        "train": {"params": {"epochs": 1}},
        "loss": {"name": "loss", "params": {}},
        "optimizer": {"name": "opt", "params": {}},
        "metrics": [{"name": "metric", "params": {}}],
        "callbacks": [{"name": "cb", "params": {}}],
        "logging": [{"name": "logger", "params": {}}],
    }


def test_validate_config_success(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_yaml", lambda _: _valid_cli_cfg())

    cli.validate_config(config_path="ok.yaml", require_experiment_name=True)
    out = capsys.readouterr().out
    assert "is valid" in out


def test_validate_config_fails_on_missing_experiment_name(monkeypatch, capsys):
    bad = _valid_cli_cfg()
    bad["experiment_info"] = {"params": {}}
    monkeypatch.setattr(cli, "load_yaml", lambda _: bad)

    with pytest.raises(typer.Exit) as exc:
        cli.validate_config(config_path="bad.yaml", require_experiment_name=True)

    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "is invalid" in out
    assert "experiment_info.params.experiment_name" in out


def test_registry_list_filters_by_type(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_available_registries",
        lambda: {
            "models": _RegistryStub(["b_model", "a_model"]),
            "losses": _RegistryStub(["mse_loss"]),
        },
    )

    cli.registry_list(kind="models")
    out = capsys.readouterr().out
    assert "models (2):" in out
    assert "- a_model" in out
    assert "- b_model" in out


def test_registry_list_unknown_type_exits(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_available_registries", lambda: {"models": _RegistryStub(["x"])})

    with pytest.raises(typer.Exit) as exc:
        cli.registry_list(kind="unknown")

    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "Unknown registry type" in out


def test_plugins_list_prints_discovered_plugins(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_resolve_entrypoint_plugins",
        lambda group: [
            _EntryPointStub("plugin_b", "pkg.b:register"),
            _EntryPointStub("plugin_a", "pkg.a:register"),
        ],
    )

    cli.plugins_list(group="chimera_ml.plugins")
    out = capsys.readouterr().out
    assert "Discovered 2 plugin(s)" in out
    assert "- plugin_a: pkg.a:register" in out
    assert "- plugin_b: pkg.b:register" in out


def test_plugins_list_handles_empty_group(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_resolve_entrypoint_plugins", lambda group: [])

    cli.plugins_list(group="chimera_ml.plugins")
    out = capsys.readouterr().out
    assert "No plugins discovered" in out


def test_doctor_prints_registry_and_plugin_counts(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_available_registries",
        lambda: {"models": _RegistryStub(["m1"]), "losses": _RegistryStub([])},
    )
    monkeypatch.setattr(
        cli,
        "_resolve_entrypoint_plugins",
        lambda group: [_EntryPointStub("p", "pkg:register")],
    )
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)

    cli.doctor(plugin_group="chimera_ml.plugins")
    out = capsys.readouterr().out
    assert "chimera-ml doctor" in out
    assert "registries:" in out
    assert "- models: 1" in out
    assert "plugins/chimera_ml.plugins: 1 discovered" in out
