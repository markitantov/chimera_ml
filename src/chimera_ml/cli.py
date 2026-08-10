import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import typer
from torch.utils.data import DataLoader

from chimera_ml.core.config import ExperimentConfig
from chimera_ml.data.loader_utils import normalize_loaders
from chimera_ml.inference import (
    InferenceConfig,
    InferenceContext,
    build_inference_pipeline,
    resolve_inference_device,
)
from chimera_ml.logging.utils import generate_run_name
from chimera_ml.training.builders import (
    BuildContext,
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
from chimera_ml.training.sweep import GridSweep, OptunaSweep
from chimera_ml.training.trainer import Trainer
from chimera_ml.utils.seed import define_seed
from chimera_ml.utils.sweep import TrainRunResult, format_sweep_overrides

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
        INFERENCE_STEPS,
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
        "inference_steps": INFERENCE_STEPS,
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


def _run_train_from_config(
    config_path: str,
    *,
    config: ExperimentConfig | None = None,
    run_name_suffix: str | None = None,
) -> TrainRunResult:
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
        base_name=experiment_info.get("run_name"),
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
    context = BuildContext(config=cfg, stage="train")

    dm = build_datamodule(cfg.section("data"), context=context)
    context.register(dm)

    model_obj = build_model(cfg.section("model"), context=context)
    context.register(model_obj)

    train_cfg = build_train_config(cfg.section("train"))

    logger_cfg = cfg.section("logging", name="console_file_logger")
    logger = None
    if logger_cfg:
        logger = build_logger(
            logger_cfg,
            inject={"experiment_name": experiment_name, "run_name": run_name},
            context=context,
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
            context=context,
        )

    loss_fn = build_loss(cfg.section("loss"), context=context)
    context.register(loss_fn)

    metrics = build_metrics(cfg.get("metrics", []), context=context)
    context.register_many(metrics)

    optimizer = build_optimizer(cfg.section("optimizer"), model_obj, context=context)
    context.register(optimizer)

    scheduler = build_scheduler(cfg.get("scheduler"), optimizer, context=context)
    if scheduler is not None:
        context.register(scheduler)

    callbacks = build_callbacks(cfg.get("callbacks"), context=context)
    context.register_many(callbacks)

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

    callbacks_cfg = cfg.get("callbacks")
    if isinstance(callbacks_cfg, list):
        for callback_cfg, callback in zip(callbacks_cfg, callbacks, strict=False):
            if callback_cfg.get("name") == "sweep_target_callback":
                return TrainRunResult(
                    run_name=run_name,
                    target_value=getattr(callback, "best_value", None),
                    target_epoch=getattr(callback, "best_epoch", None),
                )

    return TrainRunResult(run_name=run_name)


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


@app.command("inference")
def inference(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to input video/audio file."),
    output_path: str | None = typer.Option(None, "--output", "-o", help="Where to save inference JSON."),
    config_path: str = typer.Option(..., "--config-path", "-c", help="Path to inference YAML config."),
    device: str | None = typer.Option(None, "--device", help="Runtime device: cpu|cuda|auto."),
    work_dir: str | None = typer.Option(None, "--work-dir", help="Working directory for intermediate artifacts."),
):
    """Run an inference pipeline built from registry steps and YAML config."""
    typer.echo(f"[inference] Loading config: {config_path}")
    cfg = InferenceConfig.from_yaml(config_path)
    resolved_output = Path(output_path) if output_path is not None else None
    if resolved_output is not None:
        write_json_predictions_cfg = cfg.section("steps", name="write_json_predictions_step")
        if cfg.parallel and not write_json_predictions_cfg:
            typer.echo(
                "[inference] --output/-o cannot auto-create 'write_json_predictions_step' "
                "when 'pipeline.parallel: true' is enabled. Add "
                "'write_json_predictions_step' explicitly to the config and define its "
                "'after' dependencies there."
            )
            raise typer.Exit(code=1)

        existing_output_path = (
            (write_json_predictions_cfg.get("params") or {}).get("output_path") if write_json_predictions_cfg else None
        )

        if not write_json_predictions_cfg:
            typer.echo(
                "[inference] Step 'write_json_predictions_step' not found; "
                f"creating one with output_path={resolved_output}"
            )
        elif existing_output_path is not None and str(existing_output_path) != str(resolved_output):
            typer.echo(
                "[inference] write_json_predictions_step.output_path differs from config; "
                f"overriding '{existing_output_path}' -> '{resolved_output}'"
            )

        if cfg.get("steps") is None:
            cfg.raw["steps"] = []

        cfg.set_at_path("steps.write_json_predictions_step.params.output_path", str(resolved_output))

    input_file = Path(input_path)
    if not input_file.exists():
        typer.echo(f"[inference] File not found: {input_file}")
        raise typer.Exit(code=1)

    resolved_work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="chimera-inference-"))
    resolved_work_dir.mkdir(parents=True, exist_ok=True)

    runtime_device = resolve_inference_device(device or cfg.runtime_device())
    ctx = InferenceContext(
        input_path=input_file,
        work_dir=resolved_work_dir,
        device=runtime_device,
        config=cfg.raw,
    )

    pipeline = build_inference_pipeline(cfg)
    typer.echo(f"[inference] Running pipeline '{pipeline.name}' on device={runtime_device}")
    try:
        ctx = pipeline.run(ctx)
    except FileNotFoundError as exc:
        typer.echo(f"[inference] {exc}")
        raise typer.Exit(code=1) from exc

    if resolved_output is not None:
        typer.echo(f"[inference] Done. Output: {resolved_output}")
    else:
        typer.echo("[inference] Done.")


@registry_app.command("list")
def registry_list(
    kind: str | None = typer.Option(
        None,
        "--type",
        help=(
            "Filter by registry name: datamodules|models|losses|metrics|optimizers|"
            "schedulers|callbacks|collates|loggers|inference_steps."
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
    sweep_name: str | None = typer.Option(None, "--sweep-name", "-n", help="Optional human-readable sweep name."),
    max_trials: int | None = typer.Option(None, "--max-trials", help="Optional CI limit for the number of trials."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print generated trials without running training."),
):
    """Run a grid or Optuna hyperparameter sweep by repeatedly calling train."""
    base_cfg = ExperimentConfig.from_yaml(base_config)
    sweep_cfg = ExperimentConfig.from_yaml(sweep_config)

    if max_trials is not None and max_trials < 1:
        raise ValueError("--max-trials must be >= 1 when provided.")

    experiment_info = base_cfg.section("experiment_info").get("params", {})
    if "experiment_name" not in experiment_info:
        raise ValueError("`experiment_info.params.experiment_name` is required.")

    experiment_name = experiment_info["experiment_name"]
    timezone = experiment_info.get("timezone", None)
    method = str(sweep_cfg.raw.get("method", "grid")).strip().lower()
    method = "grid" if method in ("", "cartesian") else method

    if method == "grid":
        sweep_run = GridSweep(
            base_cfg=base_cfg,
            sweep_cfg=sweep_cfg,
            experiment_name=experiment_name,
            sweep_name=sweep_name,
            timezone=timezone,
            max_trials=max_trials,
        )
        total_trials = len(sweep_run.overrides)
        typer.echo(f"[sweep] Loaded {total_trials} trial(s).")

        if dry_run:
            for trial_index, overrides in enumerate(sweep_run.overrides, start=1):
                trial_id = f"{sweep_run.label}-{sweep_run.short_id}-{trial_index:03d}"
                typer.echo(
                    f"[sweep] Trial {trial_index}/{total_trials} {trial_id}: {format_sweep_overrides(overrides)}"
                )

            return

        sweep_run.start()
        try:
            for trial_index, overrides in enumerate(sweep_run.overrides, start=1):
                trial_id, trial_config_path, trial_cfg = sweep_run.write_trial_config(trial_index, overrides)
                typer.echo(
                    f"[sweep] Trial {trial_index}/{total_trials} {trial_id}: {format_sweep_overrides(overrides)}"
                )

                result = _run_train_from_config(
                    str(trial_config_path),
                    config=trial_cfg,
                    run_name_suffix=trial_id,
                )

                sweep_run.save_trial_record({"trial_id": trial_id, "run_name": result.run_name})
        except Exception:
            sweep_run.finish("failed")
            raise

        sweep_run.finish("completed")
        return

    if method == "optuna":
        sweep_run = OptunaSweep(
            base_cfg=base_cfg,
            sweep_cfg=sweep_cfg,
            experiment_name=experiment_name,
            sweep_name=sweep_name,
            timezone=timezone,
            max_trials=max_trials,
        )

        if dry_run:
            typer.echo(
                f"[sweep] Optuna dry run: {sweep_run.n_trials} trial(s), "
                f"target={sweep_run.target.monitor!r}, mode={sweep_run.target.mode!r}."
            )
            for path, spec in sweep_run.search_space.items():
                typer.echo(f"[sweep] {path}: {spec!r}")

            return

        sweep_run.start()
        study = sweep_run.create_study()
        typer.echo(f"[sweep] Running Optuna study '{sweep_run.study_name}' for {sweep_run.n_trials} trial(s).")

        def run_trial(trial: Any) -> float:
            trial_index = int(trial.number) + 1
            overrides = sweep_run.suggest_overrides(trial)

            trial_id, trial_config_path, trial_cfg = sweep_run.write_trial_config(trial_index, overrides)
            typer.echo(
                f"[sweep] Trial {trial_index}/{sweep_run.n_trials} {trial_id}: {format_sweep_overrides(overrides)}"
            )
            callbacks_cfg = trial_cfg.raw.setdefault("callbacks", [])
            if not isinstance(callbacks_cfg, list):
                raise TypeError("Config section 'callbacks' must be a list to run an Optuna sweep.")

            callbacks_cfg.append(
                {
                    "name": "sweep_target_callback",
                    "params": {"monitor": sweep_run.target.monitor, "mode": sweep_run.target.mode},
                }
            )
            trial_cfg.to_yaml(trial_config_path)

            result = _run_train_from_config(
                str(trial_config_path),
                config=trial_cfg,
                run_name_suffix=trial_id,
            )
            if result.target_value is None:
                raise ValueError(f"Sweep target '{sweep_run.target.monitor}' was not found in training logs.")

            record: dict[str, Any] = {
                "trial_id": trial_id,
                "run_name": result.run_name,
                "trial_number": int(trial.number),
                "value": result.target_value,
                "overrides": dict(overrides),
            }
            if result.target_epoch is not None:
                record["target_epoch"] = result.target_epoch

            sweep_run.save_trial_record(record)

            if hasattr(trial, "set_user_attr"):
                trial.set_user_attr("trial_id", trial_id)
                trial.set_user_attr("run_name", result.run_name)
                trial.set_user_attr("target_epoch", result.target_epoch)

            return result.target_value

        try:
            study.optimize(run_trial, n_trials=sweep_run.n_trials)
        except Exception:
            sweep_run.finish("failed")
            raise

        sweep_run.save_best_trial(study.best_trial)
        sweep_run.finish("completed")
        return

    raise ValueError(f"Unsupported sweep method '{method}'. Supported methods: grid, optuna.")


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
    context = BuildContext(config=cfg, stage="eval")

    dm = build_datamodule(cfg.section("data"), context=context)
    context.register(dm)
    model_obj = build_model(cfg.section("model"), context=context)
    context.register(model_obj)

    train_cfg = build_train_config(cfg.section("train"))
    train_cfg.epochs = 1

    loss_fn = build_loss(cfg.section("loss"), context=context)
    context.register(loss_fn)

    metrics = build_metrics(cfg.get("metrics", []), context=context)
    context.register_many(metrics)

    optimizer = build_optimizer(cfg.section("optimizer"), model_obj, context=context)
    context.register(optimizer)

    callbacks = build_callbacks(cfg.get("callbacks"), context=context)
    context.register_many(callbacks)

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
