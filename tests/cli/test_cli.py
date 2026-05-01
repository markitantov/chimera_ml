from copy import deepcopy

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


class _DMContextStub(_DMStub):
    def describe_context(self, context):
        context.set("data.inferred_dim", 11)
        context.set("data.monitor_name", f"{context.stage}/score")


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


class _ModelContextStub(_ModelStub):
    def describe_context(self, context):
        context.set("model.hidden_dim", 23)


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


def _patch_config(monkeypatch, cfg):
    monkeypatch.setattr(cli.ExperimentConfig, "from_yaml", classmethod(lambda cls, _: cls(deepcopy(cfg))))


def _patch_configs(monkeypatch, configs):
    monkeypatch.setattr(
        cli.ExperimentConfig,
        "from_yaml",
        classmethod(lambda cls, path: cls(deepcopy(configs[path]))),
    )


def test_cli_train_wires_builders_and_trainer(monkeypatch):
    model = _ModelStub()

    _patch_config(monkeypatch, _config_for_train())
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "generate_run_name", lambda **_: "run_name")
    monkeypatch.setattr(cli, "build_datamodule", lambda *args, **kwargs: _DMStub())
    monkeypatch.setattr(cli, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", lambda *args, **kwargs: "loss")
    monkeypatch.setattr(cli, "build_metrics", lambda *args, **kwargs: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *args, **kwargs: "opt")
    monkeypatch.setattr(cli, "build_scheduler", lambda *args, **kwargs: "sch")
    monkeypatch.setattr(cli, "build_callbacks", lambda *args, **kwargs: ["cb"])
    monkeypatch.setattr(
        cli,
        "build_logger",
        lambda cfg, inject=None, context=None: {"cfg": cfg, "inject": inject, "context": context},
    )
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)

    cli.train(config_path="cfg.yaml")

    assert _TrainerStub.last_init is not None
    assert _TrainerStub.last_init["scheduler"] == "sch"
    assert _TrainerStub.last_fit == ("train_loader", {"val": "val_loader"})


def test_cli_train_works_without_snapshot_callback(monkeypatch):
    model = _ModelStub()

    _patch_config(monkeypatch, _config_for_train_without_snapshot())
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "generate_run_name", lambda **_: "run_name")
    monkeypatch.setattr(cli, "build_datamodule", lambda *args, **kwargs: _DMStub())
    monkeypatch.setattr(cli, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", lambda *args, **kwargs: "loss")
    monkeypatch.setattr(cli, "build_metrics", lambda *args, **kwargs: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *args, **kwargs: "opt")
    monkeypatch.setattr(cli, "build_scheduler", lambda *args, **kwargs: "sch")
    monkeypatch.setattr(cli, "build_callbacks", lambda *args, **kwargs: ["cb"])
    monkeypatch.setattr(
        cli,
        "build_logger",
        lambda cfg, inject=None, context=None: {"cfg": cfg, "inject": inject, "context": context},
    )
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)

    cli.train(config_path="cfg.yaml")

    assert _TrainerStub.last_init is not None
    assert _TrainerStub.last_fit == ("train_loader", {"val": "val_loader"})


def test_cli_train_initializes_console_logger_before_mlflow(monkeypatch):
    model = _ModelStub()
    logger_call_order: list[str] = []

    def _build_logger(cfg, inject=None, context=None):
        logger_call_order.append(cfg["name"])
        return {"name": cfg["name"], "inject": inject, "context": context}

    _patch_config(monkeypatch, _config_for_train())
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "generate_run_name", lambda **_: "run_name")
    monkeypatch.setattr(cli, "build_datamodule", lambda *args, **kwargs: _DMStub())
    monkeypatch.setattr(cli, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", lambda *args, **kwargs: "loss")
    monkeypatch.setattr(cli, "build_metrics", lambda *args, **kwargs: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *args, **kwargs: "opt")
    monkeypatch.setattr(cli, "build_scheduler", lambda *args, **kwargs: "sch")
    monkeypatch.setattr(cli, "build_callbacks", lambda *args, **kwargs: ["cb"])
    monkeypatch.setattr(cli, "build_logger", _build_logger)
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)

    cli.train(config_path="cfg.yaml")

    assert logger_call_order == ["console_file_logger", "mlflow_logger"]
    assert _TrainerStub.last_init is not None
    assert _TrainerStub.last_init["logger"]["name"] == "console_file_logger"
    assert _TrainerStub.last_init["mlflow_logger"]["name"] == "mlflow_logger"


def test_cli_train_creates_logs_dir_before_mlflow_init(monkeypatch, tmp_path):
    model = _ModelStub()
    cfg = _config_for_train()
    cfg["logging"] = [
        {"name": "mlflow_logger", "params": {"tracking_uri": "sqlite:///logs/mlflow.db"}},
        {
            "name": "console_file_logger",
            "params": {
                "log_path": "logs",
                "log_file": "train.log",
                "console_level": "INFO",
                "file_level": "INFO",
            },
        },
    ]

    original_build_logger = cli.build_logger

    def _build_logger_wrapper(logger_cfg, inject=None, context=None):
        if logger_cfg["name"] == "mlflow_logger":
            expected_logs_dir = tmp_path / "logs" / "exp" / "run_name"
            assert expected_logs_dir.exists()
            return {"name": "mlflow_logger", "inject": inject, "context": context}

        return original_build_logger(logger_cfg, inject=inject, context=context)

    monkeypatch.chdir(tmp_path)
    _patch_config(monkeypatch, cfg)
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "generate_run_name", lambda **_: "run_name")
    monkeypatch.setattr(cli, "build_datamodule", lambda *args, **kwargs: _DMStub())
    monkeypatch.setattr(cli, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", lambda *args, **kwargs: "loss")
    monkeypatch.setattr(cli, "build_metrics", lambda *args, **kwargs: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *args, **kwargs: "opt")
    monkeypatch.setattr(cli, "build_scheduler", lambda *args, **kwargs: "sch")
    monkeypatch.setattr(cli, "build_callbacks", lambda *args, **kwargs: ["cb"])
    monkeypatch.setattr(cli, "build_logger", _build_logger_wrapper)
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)

    cli.train(config_path="cfg.yaml")

    assert (tmp_path / "logs" / "exp" / "run_name").exists()


def test_cli_train_build_context_flows_across_build_chain(monkeypatch):
    model = _ModelContextStub()
    dm = _DMContextStub()
    seen: dict[str, object] = {}

    def _build_model(cfg, *, context=None):
        seen["model_build"] = (
            context.get("data.inferred_dim"),
            context.get("model.hidden_dim"),
            context.stage,
        )
        return model

    def _build_loss(cfg, *, context=None):
        seen["loss_build"] = (
            context.get("data.inferred_dim"),
            context.get("model.hidden_dim"),
            context.stage,
        )
        return "loss"

    def _build_callbacks(cfg, *, context=None):
        seen["callback_build"] = (
            context.get("data.monitor_name"),
            context.get("model.hidden_dim"),
        )
        return ["cb"]

    _patch_config(monkeypatch, _config_for_train())
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "generate_run_name", lambda **_: "run_name")
    monkeypatch.setattr(cli, "build_datamodule", lambda *args, **kwargs: dm)
    monkeypatch.setattr(cli, "build_model", _build_model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", _build_loss)
    monkeypatch.setattr(cli, "build_metrics", lambda *args, **kwargs: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *args, **kwargs: "opt")
    monkeypatch.setattr(cli, "build_scheduler", lambda *args, **kwargs: "sch")
    monkeypatch.setattr(cli, "build_callbacks", _build_callbacks)
    monkeypatch.setattr(
        cli,
        "build_logger",
        lambda cfg, inject=None, context=None: {"cfg": cfg, "inject": inject, "context": context},
    )
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)

    cli.train(config_path="cfg.yaml")

    assert seen["model_build"] == (11, None, "train")
    assert seen["loss_build"] == (11, 23, "train")
    assert seen["callback_build"] == ("train/score", 23)


def test_cli_sweep_runs_train_for_parameter_grid(monkeypatch, tmp_path):
    base_cfg = _config_for_train()
    sweep_cfg = {
        "parameters": {
            "optimizer.params.lr": [0.001, 0.0001],
            "train.params.epochs": [1, 2],
        }
    }
    calls = []

    def _run_train(config_path, *, config=None, run_name_suffix=None):
        calls.append((config_path, config, run_name_suffix))

    _patch_configs(monkeypatch, {"base.yaml": base_cfg, "sweep.yaml": sweep_cfg})
    monkeypatch.setattr(cli, "_run_train_from_config", _run_train)

    cli.sweep(
        base_config="base.yaml",
        sweep_config="sweep.yaml",
        output_dir=str(tmp_path / "sweeps"),
        max_trials=None,
        dry_run=False,
    )

    assert len(calls) == 4
    assert calls[0][2] == "sweep_001"
    assert calls[0][1].raw["optimizer"]["params"]["lr"] == 0.001
    assert calls[0][1].raw["train"]["params"]["epochs"] == 1
    assert calls[-1][1].raw["optimizer"]["params"]["lr"] == 0.0001
    assert calls[-1][1].raw["train"]["params"]["epochs"] == 2
    assert (tmp_path / "sweeps" / "base_sweep_001.yaml").exists()
    assert (tmp_path / "sweeps" / "base_sweep_004.yaml").exists()


def test_cli_sweep_patches_named_list_sections(monkeypatch, tmp_path):
    base_cfg = _config_for_train()
    sweep_cfg = {"trials": [{"callbacks.checkpoint_callback.params.monitor": "val/ccc"}]}
    calls = []

    def _run_train(config_path, *, config=None, run_name_suffix=None):
        calls.append((config_path, config, run_name_suffix))

    _patch_configs(monkeypatch, {"base.yaml": base_cfg, "sweep.yaml": sweep_cfg})
    monkeypatch.setattr(cli, "_run_train_from_config", _run_train)

    cli.sweep(
        base_config="base.yaml",
        sweep_config="sweep.yaml",
        output_dir=str(tmp_path / "sweeps"),
        max_trials=None,
        dry_run=False,
    )

    callbacks = calls[0][1].raw["callbacks"]
    checkpoint_cfg = next(item for item in callbacks if item["name"] == "checkpoint_callback")
    assert checkpoint_cfg["params"]["monitor"] == "val/ccc"


def test_cli_eval_loads_checkpoint_and_calls_evaluate(monkeypatch):
    model = _ModelStub()

    _patch_config(monkeypatch, _config_for_eval())
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "build_datamodule", lambda *args, **kwargs: _DMEvalStub())
    monkeypatch.setattr(cli, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", lambda *args, **kwargs: "loss")
    monkeypatch.setattr(cli, "build_metrics", lambda *args, **kwargs: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *args, **kwargs: "opt")
    monkeypatch.setattr(cli, "build_callbacks", lambda *args, **kwargs: ["cb"])
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)
    monkeypatch.setattr(cli.torch, "load", lambda *args, **kwargs: {"model_state_dict": {}})

    cli.eval(
        config_path="cfg_eval.yaml",
        checkpoint_path="ckpt.pt",
        with_features=True,
    )

    assert _TrainerStub.last_init is not None
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

    _patch_config(monkeypatch, _config_for_eval())
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "build_datamodule", lambda *args, **kwargs: _DMNestedStub())
    monkeypatch.setattr(cli, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", lambda *args, **kwargs: "loss")
    monkeypatch.setattr(cli, "build_metrics", lambda *args, **kwargs: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *args, **kwargs: "opt")
    monkeypatch.setattr(cli, "build_callbacks", lambda *args, **kwargs: ["cb"])
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)
    monkeypatch.setattr(cli.torch, "load", lambda *args, **kwargs: {"model_state_dict": {}})

    cli.eval(
        config_path="cfg_eval.yaml",
        checkpoint_path="ckpt.pt",
        with_features=False,
    )

    assert _TrainerStub.last_eval is not None
    loaders, with_features, feature_extractor = _TrainerStub.last_eval
    assert set(loaders.keys()) == {"train_main", "val0", "val1", "test_a"}
    assert with_features is False
    assert feature_extractor is None


def test_cli_eval_uses_weights_only_when_loading_checkpoint(monkeypatch):
    model = _ModelStub()
    seen_kwargs: dict[str, object] = {}

    def _fake_torch_load(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return {"model_state_dict": {}}

    _patch_config(monkeypatch, _config_for_eval())
    monkeypatch.setattr(cli, "define_seed", lambda _: None)
    monkeypatch.setattr(cli, "build_datamodule", lambda *args, **kwargs: _DMEvalStub())
    monkeypatch.setattr(cli, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(cli, "build_train_config", lambda _: _TrainCfg())
    monkeypatch.setattr(cli, "build_loss", lambda *args, **kwargs: "loss")
    monkeypatch.setattr(cli, "build_metrics", lambda *args, **kwargs: ["metric"])
    monkeypatch.setattr(cli, "build_optimizer", lambda *args, **kwargs: "opt")
    monkeypatch.setattr(cli, "build_callbacks", lambda *args, **kwargs: ["cb"])
    monkeypatch.setattr(cli, "Trainer", _TrainerStub)
    monkeypatch.setattr(cli.torch, "load", _fake_torch_load)

    cli.eval(
        config_path="cfg_eval.yaml",
        checkpoint_path="ckpt.pt",
        with_features=False,
    )

    assert seen_kwargs.get("weights_only") is True


def test_cli_inference_builds_pipeline_and_runs(monkeypatch, tmp_path, capsys):
    seen: dict[str, object] = {}

    class _PipelineStub:
        name = "demo_inference"

        def run(self, ctx):
            seen["ctx"] = ctx
            return ctx

    monkeypatch.setattr(
        cli.InferenceConfig,
        "from_yaml",
        classmethod(lambda cls, _: cls({"pipeline": {"name": "demo_inference"}, "steps": []})),
    )
    monkeypatch.setattr(cli, "build_inference_pipeline", lambda cfg: _PipelineStub())
    monkeypatch.setattr(cli, "resolve_inference_device", lambda _: "cpu")

    output_path = tmp_path / "out.json"
    work_dir = tmp_path / "work"
    cli.inference(
        input_path="video.mp4",
        output_path=str(output_path),
        config_path="inference.yaml",
        device="auto",
        work_dir=str(work_dir),
    )

    ctx = seen["ctx"]
    assert ctx.input_path.name == "video.mp4"
    assert ctx.work_dir == work_dir
    assert ctx.device == "cpu"
    assert ctx.config["steps"][-1] == {
        "name": "write_json_predictions_step",
        "params": {"output_path": str(output_path)},
    }
    assert "Step 'write_json_predictions_step' not found; creating one" in capsys.readouterr().out


def test_cli_inference_overrides_write_json_path(monkeypatch, tmp_path, capsys):
    seen: dict[str, object] = {}

    class _PipelineStub:
        name = "demo_inference"

        def run(self, ctx):
            seen["ctx"] = ctx
            return ctx

    monkeypatch.setattr(
        cli.InferenceConfig,
        "from_yaml",
        classmethod(
            lambda cls, _: cls(
                {
                    "pipeline": {"name": "demo_inference"},
                    "steps": [{"name": "write_json_predictions_step", "params": {"output_path": "old.json"}}],
                }
            )
        ),
    )
    monkeypatch.setattr(cli, "build_inference_pipeline", lambda cfg: _PipelineStub())
    monkeypatch.setattr(cli, "resolve_inference_device", lambda _: "cpu")

    output_path = tmp_path / "out.json"
    cli.inference(
        input_path="video.mp4",
        output_path=str(output_path),
        config_path="inference.yaml",
        device="auto",
        work_dir=str(tmp_path / "work"),
    )

    out = capsys.readouterr().out
    assert seen["ctx"].config["steps"] == [
        {"name": "write_json_predictions_step", "params": {"output_path": str(output_path)}}
    ]
    assert "overriding 'old.json'" in out


def test_cli_inference_leaves_config_unchanged_without_output_override(monkeypatch, tmp_path, capsys):
    seen: dict[str, object] = {}

    class _PipelineStub:
        name = "demo_inference"

        def run(self, ctx):
            seen["ctx"] = ctx
            return ctx

    monkeypatch.setattr(
        cli.InferenceConfig,
        "from_yaml",
        classmethod(
            lambda cls, _: cls(
                {
                    "pipeline": {"name": "demo_inference"},
                    "steps": [{"name": "print_json_predictions_step", "params": {}}],
                }
            )
        ),
    )
    monkeypatch.setattr(cli, "build_inference_pipeline", lambda cfg: _PipelineStub())
    monkeypatch.setattr(cli, "resolve_inference_device", lambda _: "cpu")

    cli.inference(
        input_path="video.mp4",
        output_path=None,
        config_path="inference.yaml",
        device="auto",
        work_dir=str(tmp_path / "work"),
    )

    ctx = seen["ctx"]
    assert ctx.config["steps"] == [{"name": "print_json_predictions_step", "params": {}}]
    assert "creating one" not in capsys.readouterr().out


def test_cli_inference_creates_steps_section_when_missing(monkeypatch, tmp_path, capsys):
    seen: dict[str, object] = {}

    class _PipelineStub:
        name = "demo_inference"

        def run(self, ctx):
            seen["ctx"] = ctx
            return ctx

    monkeypatch.setattr(
        cli.InferenceConfig,
        "from_yaml",
        classmethod(lambda cls, _: cls({"pipeline": {"name": "demo_inference"}})),
    )
    monkeypatch.setattr(cli, "build_inference_pipeline", lambda cfg: _PipelineStub())
    monkeypatch.setattr(cli, "resolve_inference_device", lambda _: "cpu")

    output_path = tmp_path / "out.json"
    cli.inference(
        input_path="video.mp4",
        output_path=str(output_path),
        config_path="inference.yaml",
        device="auto",
        work_dir=str(tmp_path / "work"),
    )

    assert seen["ctx"].config["steps"] == [
        {"name": "write_json_predictions_step", "params": {"output_path": str(output_path)}}
    ]
    assert "Step 'write_json_predictions_step' not found; creating one" in capsys.readouterr().out


def test_cli_inference_creates_temp_work_dir_when_missing(monkeypatch, tmp_path, capsys):
    seen: dict[str, object] = {}

    class _PipelineStub:
        name = "demo_inference"

        def run(self, ctx):
            seen["ctx"] = ctx
            return ctx

    generated_work_dir = tmp_path / "generated-work-dir"
    monkeypatch.setattr(
        cli.InferenceConfig,
        "from_yaml",
        classmethod(lambda cls, _: cls({"pipeline": {"name": "demo_inference"}, "steps": []})),
    )
    monkeypatch.setattr(cli, "build_inference_pipeline", lambda cfg: _PipelineStub())
    monkeypatch.setattr(cli, "resolve_inference_device", lambda _: "cpu")
    monkeypatch.setattr(cli.tempfile, "mkdtemp", lambda prefix: str(generated_work_dir))

    cli.inference(
        input_path="video.mp4",
        output_path=None,
        config_path="inference.yaml",
        device="auto",
        work_dir=None,
    )

    ctx = seen["ctx"]
    assert ctx.work_dir == generated_work_dir
    assert generated_work_dir.exists()
    assert "[inference] Done." in capsys.readouterr().out


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
    _patch_config(monkeypatch, _valid_cli_cfg())

    cli.validate_config(config_path="ok.yaml", require_experiment_name=True)
    out = capsys.readouterr().out
    assert "is valid" in out


def test_validate_config_fails_on_missing_experiment_name(monkeypatch, capsys):
    bad = _valid_cli_cfg()
    bad["experiment_info"] = {"params": {}}
    _patch_config(monkeypatch, bad)

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
