import platform
import sys
from collections.abc import Iterator, Mapping, Sequence
from itertools import product
from pathlib import Path
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
from chimera_ml.training.config import ExperimentConfig
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


def _iter_sweep_overrides(sweep_cfg: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield trial override dictionaries from explicit trials or a grid parameter spec."""
    trials = sweep_cfg.get("trials")
    parameters = sweep_cfg.get("parameters")

    if trials is not None and parameters is not None:
        raise ValueError("Sweep config must define either 'trials' or 'parameters', not both.")

    if trials is not None:
        if not isinstance(trials, Sequence) or isinstance(trials, (str, bytes)):
            raise TypeError("Sweep config 'trials' must be a list of mappings.")

        for i, trial in enumerate(trials, start=1):
            if not isinstance(trial, Mapping):
                raise TypeError(f"Sweep trial #{i} must be a mapping.")

            yield dict(trial)

        return

    if not isinstance(parameters, Mapping):
        raise TypeError("Sweep config must contain a 'parameters' mapping or a 'trials' list.")

    paths = list(parameters.keys())
    value_lists: list[list[Any]] = []
    for path in paths:
        values = parameters[path]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"Sweep parameter '{path}' must be a non-empty list.")

        if not values:
            raise ValueError(f"Sweep parameter '{path}' must contain at least one value.")

        value_lists.append(list(values))

    for values in product(*value_lists):
        yield dict(zip(paths, values, strict=True))


def _format_overrides(overrides: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in overrides.items())


def _run_train_from_config(
    config_path: str,
    *,
    config: ExperimentConfig | None = None,
    run_name_suffix: str | None = None,
) -> None:
    """Run training from a config path or an already loaded config dict."""
    typer.echo(f"[train] Loading config: {config_path}")
    cfg = config.copy() if config is not None else ExperimentConfig.from_yaml(config_path)

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
        suffix=run_name_suffix,
        include_time=experiment_info.get("include_time", True),
        datetime_format=experiment_info.get("datetime_format", "%Y-%m-%d_%H-%M"),
        timezone=experiment_info.get("timezone", None),
    )
    typer.echo(f"[train] Experiment: {experiment_name} | Run: {run_name}")

    if isinstance(cfg.get("callbacks"), list) and cfg.section("callbacks", name="checkpoint_callback"):
        cfg.set_at_path("callbacks.checkpoint_callback.params.experiment_name", experiment_name)
        cfg.set_at_path("callbacks.checkpoint_callback.params.run_name", run_name)

    snapshot_cfg = cfg.section("callbacks", name="snapshot_callback")
    snapshot_params = snapshot_cfg.get("params", {}) if snapshot_cfg else {}
    save_snapshot_config = bool(snapshot_params.get("save_config"))

    if isinstance(cfg.get("callbacks"), list) and snapshot_cfg:
        cfg.set_at_path("callbacks.snapshot_callback.params.experiment_name", experiment_name)
        cfg.set_at_path("callbacks.snapshot_callback.params.run_name", run_name)
        cfg.set_at_path("callbacks.snapshot_callback.params.config_path", config_path if save_snapshot_config else None)

    # 1) Load project plugins (register datamodule/model/loss/metrics/callbacks/etc)

    # 2) Build from registries
    typer.echo("[train] Building datamodule and model...")
    dm = build_datamodule(cfg.section("data"))
    model_obj = build_model(cfg.section("model"))

    train_cfg = build_train_config(cfg.section("train"))

    logger_cfg = cfg.section("logging", name="console_file_logger")
    logger = None
    if logger_cfg:
        logger = build_logger(logger_cfg, inject={"experiment_name": experiment_name, "run_name": run_name})

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


@app.command("validate-config")
def validate_config(
    config_path: str = typer.Option(..., "--config-path", "-c", help="Path to experiment YAML config."),
    require_experiment_name: bool = typer.Option(
        True,
        "--require-experiment-name/--no-require-experiment-name",
        help="Require `experiment_info.params.experiment_name`.",
    ),
):
    """Validate YAML config structure without starting training."""
    try:
        cfg = ExperimentConfig.from_yaml(config_path)
    except Exception as exc:
        typer.echo(f"Invalid config file '{config_path}': {exc}")
        raise typer.Exit(code=1) from exc

    errors = cfg.validate(require_experiment_name=require_experiment_name)
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
    group: str = typer.Option("chimera_ml.plugins", "--group", help="Entry point group to inspect."),
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
    config_path: str = typer.Option(..., "--config-path", "-c", help="Path to experiment YAML config."),
):
    """Run training from YAML config with dynamic factories."""
    _run_train_from_config(config_path)


@app.command()
def sweep(
    base_config: str = typer.Option(..., "--base-config", "-b", help="Path to base experiment YAML config."),
    sweep_config: str = typer.Option(..., "--sweep-config", "-s", help="Path to sweep YAML config."),
    output_dir: str = typer.Option(
        "sweep_runs",
        "--output-dir",
        "-o",
        help="Directory for materialized trial YAML files.",
    ),
    max_trials: int | None = typer.Option(None, "--max-trials", help="Optional CI limit for the number of trials."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print generated trials without running training."),
):
    """Run a grid/explicit hyperparameter sweep by repeatedly calling train."""
    base_cfg = ExperimentConfig.from_yaml(base_config)
    sweep_cfg = ExperimentConfig.from_yaml(sweep_config)

    if max_trials is not None and max_trials < 1:
        raise ValueError("--max-trials must be >= 1 when provided.")

    overrides_list = list(_iter_sweep_overrides(sweep_cfg.raw))
    if max_trials is not None:
        overrides_list = overrides_list[:max_trials]

    if not overrides_list:
        raise ValueError("Sweep config produced no trials.")

    out_path = Path(output_dir)
    if not dry_run:
        out_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[sweep] Loaded {len(overrides_list)} trial(s).")
    for i, overrides in enumerate(overrides_list, start=1):
        trial_id = f"sweep_{i:03d}"
        trial_cfg = base_cfg.copy()
        trial_cfg.apply_overrides(overrides)

        trial_config_path = out_path / f"{Path(base_config).stem}_{trial_id}.yaml"
        typer.echo(f"[sweep] Trial {i}/{len(overrides_list)} {trial_id}: {_format_overrides(overrides)}")

        if dry_run:
            continue

        trial_cfg.to_yaml(trial_config_path)
        _run_train_from_config(str(trial_config_path), config=trial_cfg, run_name_suffix=trial_id)


@app.command()
def eval(
    config_path: str = typer.Option(..., "--config-path", "-c", help="Path to experiment YAML config."),
    checkpoint_path: str = typer.Option(
        ..., "--checkpoint-path", help="Path to .pt checkpoint saved by ModelCheckpoint."
    ),
    with_features: bool | None = typer.Option(None, "--with-features"),
):
    """Run evaluation (no training). Logs metrics and artifacts to MLflow if configured."""
    typer.echo(f"[eval] Loading config: {config_path}")
    cfg = ExperimentConfig.from_yaml(config_path)

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
