import importlib
from typing import Any

import torch
import typer
from torch.utils.data import DataLoader

from chimera_ml.data.loader_utils import normalize_loaders
from chimera_ml.logging.utils import generate_run_name
from chimera_ml.training.builders import (
    build_callbacks,
    build_datamodule,
    build_logger,
    build_loss,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
    build_train_config,
)
from chimera_ml.training.config import ExperimentConfig, load_yaml
from chimera_ml.training.trainer import Trainer
from chimera_ml.utils.seed import define_seed

app = typer.Typer(add_completion=False)


def import_object(spec: str) -> Any:
    """Import object by 'module.submodule:attr'."""
    if ":" not in spec:
        raise ValueError("Import spec must be in format 'module:attr'")
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def _merge_eval_loaders(dm: Any) -> dict[str, DataLoader]:
    """Merge train/val/test dataloaders into a flat split->loader mapping for evaluation."""
    merged: dict[str, DataLoader] = {}

    for prefix, raw_loaders in (
        ("train", dm.train_dataloader()),
        ("val", dm.val_dataloader()),
        ("test", dm.test_dataloader()),
    ):
        normalized = normalize_loaders(raw_loaders, default_name=prefix)
        for name, loader in normalized.items():
            base_key = name if (name == prefix or name.startswith(prefix)) else f"{prefix}_{name}"
            key = base_key
            i = 2
            while key in merged:
                key = f"{base_key}_{i}"
                i += 1

            merged[key] = loader

    return merged


@app.command()
def train(
    config_path: str = typer.Option(..., "--config-path", "-c", help="Path to experiment YAML config."),
    class_names: str | None = typer.Option(None, "--class-names", help="Comma-separated class names."),
):
    """Run training from YAML config with dynamic factories."""
    cfg = ExperimentConfig(load_yaml(config_path))

    seed = int(cfg.get("seed", 0))
    define_seed(seed)

    # 0) Working with experiment names
    experiment_info = cfg.section("experiment_info").get("params", {})
    if "experiment_name" not in experiment_info:
        raise ValueError("`experiment_info.params.experiment_name` is required.")
    
    experiment_name = experiment_info["experiment_name"]
    run_name = generate_run_name(
        config_path=config_path,
        model_name=cfg.section("model").get("name"),
        include_time=experiment_info.get("include_time", True),
        datetime_format=experiment_info.get("datetime_format", "%Y-%m-%d_%H-%M"),
        timezone=experiment_info.get("timezone", None),
    )

    cfg.patch_params_at(
        "callbacks",
        names=["checkpoint_callback"],
        experiment_name=experiment_name,
        run_name=run_name,
    )

    snapshot_cfg = cfg.section("callbacks", name="snapshot_callback")
    snapshot_params = snapshot_cfg.get("params", {}) if snapshot_cfg else {}
    save_snapshot_config = bool(snapshot_params.get("save_config"))

    cfg.patch_params_at(
        "callbacks",
        names=["snapshot_callback"],
        experiment_name=experiment_name,
        run_name=run_name,
        config_path=config_path if save_snapshot_config else None,
    )

    # 1) Load project plugins (register datamodule/model/loss/metrics/callbacks/etc)

    # 2) Build from registries
    dm = build_datamodule(cfg.section("data"))
    model_obj = build_model(cfg.section("model"))

    train_cfg = build_train_config(cfg.section("train"))

    mlflow_cfg = cfg.section("logging", name="mlflow_logger")
    mlflow_logger = None
    if mlflow_cfg:
        mlflow_logger = build_logger(
            mlflow_cfg,
            inject={
                "config_path": config_path,
                "experiment_name": experiment_name,
                "run_name": run_name,
            },
        )

    logger_cfg = cfg.section("logging", name="console_file_logger")
    logger = None
    if logger_cfg:
        logger = build_logger(logger_cfg, inject={"experiment_name": experiment_name, "run_name": run_name})

    loss_fn = build_loss(cfg.section("loss"))
    metrics = build_metrics(cfg.get("metrics", []))

    optimizer = build_optimizer(cfg.section("optimizer"), model_obj)
    scheduler = build_scheduler(cfg.get("scheduler"), optimizer)
    callbacks = build_callbacks(cfg.get("callbacks"))

    cn = [x.strip() for x in class_names.split(",")] if class_names else None

    trainer = Trainer(
        model=model_obj,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        config=train_cfg,
        mlflow_logger=mlflow_logger,
        logger=logger,
        class_names=cn,
        callbacks=callbacks,
        scheduler=scheduler,
    )

    train_loader = dm.train_dataloader()
    val_loaders = dm.val_dataloader()

    trainer.fit(train_loader, val_loaders=val_loaders)


@app.command()
def eval(
    config_path: str = typer.Option(..., "--config-path", "-c", help="Path to experiment YAML config."),
    checkpoint_path: str | None = typer.Option(None, "--checkpoint-path", help="Path to .pt checkpoint saved by ModelCheckpoint."),
    with_features: bool | None = typer.Option(None, "--with-features"),
    class_names: str | None = typer.Option(None, "--class-names", help="Comma-separated class names."),
):
    """Run evaluation (no training). Logs metrics and artifacts to MLflow if configured."""
    cfg = ExperimentConfig(load_yaml(config_path))

    seed = int(cfg.get("seed", 0))
    define_seed(seed)

    # 1) Load project plugins (register datamodule/model/loss/metrics/callbacks/etc)
    dm = build_datamodule(cfg.section("data"))
    model_obj = build_model(cfg.section("model"))

    train_cfg = build_train_config(cfg.section("train"))
    train_cfg.epochs = 1

    loss_fn = build_loss(cfg.section("loss"))
    metrics = build_metrics(cfg.get("metrics", []))

    optimizer = build_optimizer(cfg.section("optimizer"), model_obj)
    callbacks = build_callbacks(cfg.get("callbacks"))

    cn = [x.strip() for x in class_names.split(",")] if class_names else None

    trainer = Trainer(
        model=model_obj,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        config=train_cfg,
        mlflow_logger=None,
        logger=None,
        class_names=cn,
        callbacks=callbacks,
        scheduler=None,
    )

    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu")
        state = payload.get("model_state_dict", payload)
        trainer.model.load_state_dict(state, strict=True)

    loaders = _merge_eval_loaders(dm)
    if not loaders:
        raise ValueError("No dataloaders available for evaluation.")

    trainer.evaluate(
        loaders,
        with_features=bool(with_features),
        feature_extractor=None,
    )
