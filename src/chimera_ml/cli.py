import platform
import sys
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
registry_app = typer.Typer(help="Inspect registered components.")
plugins_app = typer.Typer(help="Inspect entry-point plugins.")
app.add_typer(registry_app, name="registry")
app.add_typer(plugins_app, name="plugins")


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


def _available_registries() -> dict[str, Any]:
    """Return known registry objects keyed by CLI name."""
    from chimera_ml.core.registry import (
        CALLBACKS,
        COLLATES,
        DATAMODULES,
        LOGGERS,
        LOSSES,
        METRICS,
        MODELS,
        OPTIMIZERS,
        SCHEDULERS,
    )

    return {
        "datamodules": DATAMODULES,
        "models": MODELS,
        "losses": LOSSES,
        "metrics": METRICS,
        "optimizers": OPTIMIZERS,
        "schedulers": SCHEDULERS,
        "callbacks": CALLBACKS,
        "collates": COLLATES,
        "loggers": LOGGERS,
    }


def _resolve_entrypoint_plugins(group: str = "chimera_ml.plugins") -> list[Any]:
    """Return discovered Python entry points for the given plugin group."""
    try:
        from importlib.metadata import entry_points  # py3.10+
    except Exception:  # pragma: no cover
        try:
            from importlib_metadata import entry_points  # type: ignore
        except Exception:
            return []

    try:
        eps = entry_points()
        if hasattr(eps, "select"):
            return list(eps.select(group=group))

        return list(eps.get(group, []))  # type: ignore[attr-defined]
    except Exception:
        return []


def _collect_config_errors(
    cfg: ExperimentConfig,
    *,
    require_experiment_name: bool = True,
) -> list[str]:
    """Return human-readable config validation errors."""
    errors: list[str] = []
    raw = cfg.raw

    if not isinstance(raw, dict):
        return ["Top-level config must be a mapping/object."]

    # Named sections required by current train/eval builders.
    for section in ("data", "model", "train", "loss", "optimizer"):
        value = raw.get(section)
        if not isinstance(value, dict):
            errors.append(f"Section '{section}' must be a mapping.")
            continue

        if section != "train" and not value.get("name"):
            errors.append(f"Section '{section}' must contain non-empty 'name'.")

        params = value.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            errors.append(f"Section '{section}.params' must be a mapping.")

    scheduler = raw.get("scheduler")
    if scheduler is not None:
        if not isinstance(scheduler, dict):
            errors.append("Section 'scheduler' must be a mapping when provided.")
        elif not scheduler.get("name"):
            errors.append("Section 'scheduler' must contain non-empty 'name' when provided.")

    for section in ("metrics", "callbacks", "logging"):
        value = raw.get(section)
        if value is None:
            continue

        if not isinstance(value, list):
            errors.append(f"Section '{section}' must be a list.")
            continue

        for i, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"Section '{section}[{i}]' must be a mapping.")
                continue

            if not item.get("name"):
                errors.append(f"Section '{section}[{i}]' must contain non-empty 'name'.")

    if require_experiment_name:
        experiment_info = raw.get("experiment_info")
        if not isinstance(experiment_info, dict):
            errors.append("Section 'experiment_info' must be a mapping.")
        else:
            params = experiment_info.get("params", {})
            if not isinstance(params, dict):
                errors.append("Section 'experiment_info.params' must be a mapping.")

            elif not params.get("experiment_name"):
                errors.append("`experiment_info.params.experiment_name` is required.")

    return errors


@app.command("validate-config")
def validate_config(
    config_path: str = typer.Option(
        ..., "--config-path", "-c", help="Path to experiment YAML config."
    ),
    require_experiment_name: bool = typer.Option(
        True,
        "--require-experiment-name/--no-require-experiment-name",
        help="Require `experiment_info.params.experiment_name`.",
    ),
):
    """Validate YAML config structure without starting training."""
    try:
        cfg = ExperimentConfig(load_yaml(config_path))
    except Exception as exc:
        typer.echo(f"Invalid config file '{config_path}': {exc}")
        raise typer.Exit(code=1) from exc

    errors = _collect_config_errors(cfg, require_experiment_name=require_experiment_name)
    if errors:
        typer.echo(f"Config '{config_path}' is invalid:")
        for err in errors:
            typer.echo(f"- {err}")

        raise typer.Exit(code=1)

    typer.echo(f"Config '{config_path}' is valid.")


@registry_app.command("list")
def registry_list(
    kind: str | None = typer.Option(
        None,
        "--type",
        help=(
            "Filter by registry name: datamodules|models|losses|metrics|optimizers|"
            "schedulers|callbacks|collates|loggers."
        ),
    ),
):
    """List registered component keys."""
    registries = _available_registries()

    if kind is not None:
        key = kind.lower().strip()
        if key not in registries:
            known = ", ".join(sorted(registries.keys()))
            typer.echo(f"Unknown registry type '{kind}'. Known: {known}")
            raise typer.Exit(code=1)

        keys = registries[key].keys()
        typer.echo(f"{key} ({len(keys)}):")
        for item in keys:
            typer.echo(f"- {item}")

        return

    for reg_name in sorted(registries.keys()):
        keys = registries[reg_name].keys()
        typer.echo(f"{reg_name} ({len(keys)}):")
        for item in keys:
            typer.echo(f"- {item}")

        typer.echo("")


@plugins_app.command("list")
def plugins_list(
    group: str = typer.Option(
        "chimera_ml.plugins", "--group", help="Entry point group to inspect."
    ),
):
    """List discovered plugin entry points."""
    plugins = _resolve_entrypoint_plugins(group)
    if not plugins:
        typer.echo(f"No plugins discovered in '{group}'.")
        return

    typer.echo(f"Discovered {len(plugins)} plugin(s) in '{group}':")
    for ep in sorted(plugins, key=lambda x: getattr(x, "name", "")):
        name = getattr(ep, "name", "unknown")
        value = getattr(ep, "value", None)
        if not value:
            module = getattr(ep, "module", None)
            attr = getattr(ep, "attr", None)
            if module and attr:
                value = f"{module}:{attr}"
            elif module:
                value = str(module)
            else:
                value = "<unknown>"

        typer.echo(f"- {name}: {value}")


@app.command()
def doctor(
    plugin_group: str = typer.Option("chimera_ml.plugins", "--plugin-group"),
):
    """Print quick environment diagnostics useful for support/debugging."""
    typer.echo("chimera-ml doctor")
    typer.echo(f"python: {sys.version.split()[0]}")
    typer.echo(f"platform: {platform.platform()}")
    typer.echo(f"torch: {torch.__version__}")
    typer.echo(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(f"cuda_device_count: {torch.cuda.device_count()}")

    try:
        import mlflow  # type: ignore

        typer.echo(f"mlflow: {getattr(mlflow, '__version__', 'unknown')}")
    except Exception:
        typer.echo("mlflow: not installed")

    registries = _available_registries()
    typer.echo("registries:")
    for name in sorted(registries.keys()):
        typer.echo(f"- {name}: {len(registries[name].keys())}")

    plugins = _resolve_entrypoint_plugins(plugin_group)
    typer.echo(f"plugins/{plugin_group}: {len(plugins)} discovered")


@app.command()
def train(
    config_path: str = typer.Option(
        ..., "--config-path", "-c", help="Path to experiment YAML config."
    ),
):
    """Run training from YAML config with dynamic factories."""
    typer.echo(f"[train] Loading config: {config_path}")
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
    typer.echo(f"[train] Experiment: {experiment_name} | Run: {run_name}")

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
    typer.echo("[train] Building datamodule and model...")
    dm = build_datamodule(cfg.section("data"))
    model_obj = build_model(cfg.section("model"))

    train_cfg = build_train_config(cfg.section("train"))

    logger_cfg = cfg.section("logging", name="console_file_logger")
    logger = None
    if logger_cfg:
        logger = build_logger(
            logger_cfg, inject={"experiment_name": experiment_name, "run_name": run_name}
        )

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

    loss_fn = build_loss(cfg.section("loss"))
    metrics = build_metrics(cfg.get("metrics", []))

    optimizer = build_optimizer(cfg.section("optimizer"), model_obj)
    scheduler = build_scheduler(cfg.get("scheduler"), optimizer)
    callbacks = build_callbacks(cfg.get("callbacks"))

    trainer = Trainer(
        model=model_obj,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        config=train_cfg,
        mlflow_logger=mlflow_logger,
        logger=logger,
        callbacks=callbacks,
        scheduler=scheduler,
    )

    train_loader = dm.train_dataloader()
    val_loaders = dm.val_dataloader()

    typer.echo("[train] Starting fit...")
    trainer.fit(train_loader, val_loaders=val_loaders)
    typer.echo("[train] Done.")


@app.command()
def eval(
    config_path: str = typer.Option(
        ..., "--config-path", "-c", help="Path to experiment YAML config."
    ),
    checkpoint_path: str = typer.Option(
        ..., "--checkpoint-path", help="Path to .pt checkpoint saved by ModelCheckpoint."
    ),
    with_features: bool | None = typer.Option(None, "--with-features"),
):
    """Run evaluation (no training). Logs metrics and artifacts to MLflow if configured."""
    typer.echo(f"[eval] Loading config: {config_path}")
    cfg = ExperimentConfig(load_yaml(config_path))

    seed = int(cfg.get("seed", 0))
    define_seed(seed)

    # 1) Load project plugins (register datamodule/model/loss/metrics/callbacks/etc)
    typer.echo("[eval] Building datamodule and model...")
    dm = build_datamodule(cfg.section("data"))
    model_obj = build_model(cfg.section("model"))

    train_cfg = build_train_config(cfg.section("train"))
    train_cfg.epochs = 1

    loss_fn = build_loss(cfg.section("loss"))
    metrics = build_metrics(cfg.get("metrics", []))

    optimizer = build_optimizer(cfg.section("optimizer"), model_obj)
    callbacks = build_callbacks(cfg.get("callbacks"))

    trainer = Trainer(
        model=model_obj,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        config=train_cfg,
        mlflow_logger=None,
        logger=None,
        callbacks=callbacks,
        scheduler=None,
    )

    typer.echo(f"[eval] Loading checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload)
    trainer.model.load_state_dict(state, strict=True)

    loaders = _merge_eval_loaders(dm)
    if not loaders:
        raise ValueError("No dataloaders available for evaluation.")

    typer.echo(f"[eval] Running evaluation on {len(loaders)} loader(s): {', '.join(sorted(loaders))}")
    trainer.evaluate(
        loaders,
        with_features=bool(with_features),
        feature_extractor=None,
    )
    typer.echo("[eval] Done.")
