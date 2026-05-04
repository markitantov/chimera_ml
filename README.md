# Chimera ML

`Chimera ML (Cross-modal Hierarchical Merging of Embeddings and Representations)` is a lightweight framework for training and evaluating configurable uni-modal and multi-modal models.

The core idea is simple:

- the library provides runtime infrastructure (trainer, logging, callbacks, registry builders),
- task-specific components (datamodules, models, losses, metrics, callbacks) live in external plugin packages,
- experiments are driven by YAML configs and CLI commands.

## Requirements

- Python `>=3.12,<3.13`
- PyTorch `>=2.2,<3.0`

## Installation

Install from PyPI:

```bash
pip install chimera-ml
```

or:

```bash
poetry add chimera-ml
```

By default, the package includes MLflow, matplotlib, and requests dependencies.

Install from source (development):

```bash
poetry install --with dev
```

CLI entry points:

```bash
chimera-ml --help
chimera_ml --help
```

Both names are available; docs use `chimera-ml`.

## Quick Start

1. Install `chimera-ml`.
2. Install your task plugin package (example below).
3. Run train/eval with YAML config.

Example plugin from this repo:

```bash
pip install -e examples/va_estimation
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml train --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml eval --config-path examples/va_estimation/configs/multimodal_test.yaml --checkpoint-path path/to/last.pt
```

## CLI

Main commands:

```bash
chimera-ml validate-config --config-path <config.yaml>
chimera-ml doctor
chimera-ml train --config-path <config.yaml>
chimera-ml sweep --base-config <config.yaml> --sweep-config <sweep.yaml> [--max-trials N]
chimera-ml eval --config-path <config.yaml> --checkpoint-path <ckpt.pt> [--with-features]
chimera-ml inference -i <input.mp4> [-o <out.json>] --config-path <inference.yaml> [--device cpu|cuda|auto] [--work-dir <dir>]
chimera-ml registry list [--type models|losses|metrics|optimizers|schedulers|callbacks|collates|loggers|datamodules|inference_steps]
chimera-ml plugins list [--group chimera_ml.plugins]
```

`validate-config`:

- checks config structure and required sections without starting training.

`doctor`:

- prints quick environment diagnostics (Python, torch, CUDA, MLflow, registry/plugin counts).

`train`:

- requires `experiment_info.params.experiment_name`,
- generates `run_name` via `generate_run_name(...)`,
- patches `checkpoint_callback` and `snapshot_callback` params with experiment/run data,
- builds all components from registries and runs `Trainer.fit(...)`.

`sweep`:

- materializes one YAML config per trial under `sweep_runs/` by default,
- supports Cartesian grids via `parameters` (grid search) and explicit trial lists via `trials`,
- applies overrides using dotted paths such as `optimizer.params.lr` or
  `callbacks.checkpoint_callback.params.monitor`,
- runs the normal `train` flow once per trial and appends `sweep_001`, `sweep_002`, ... to run names,
- supports `--max-trials` for CI smoke tuning and `--dry-run` to inspect generated trials.

`eval`:

- builds datamodule/model/loss/metrics/callbacks from the same config style,
- optionally loads checkpoint (`model_state_dict` or raw state dict),
- evaluates over merged `train`/`val`/`test` loader splits when available.

`inference`:

- loads a small inference pipeline from YAML,
- builds steps from the `inference_steps` registry,
- runs them on a shared `InferenceContext`,
- runs sequentially by default in plain config order,
- enables DAG/parallel scheduling only when `pipeline.parallel: true` is set,
- uses `after` for explicit dependencies between steps in parallel mode,
- keeps output behavior inside explicit pipeline steps such as `write_json_predictions_step` and `print_json_predictions_step`.
- supports a built-in `resolve_checkpoints_step` that resolves local paths or downloads remote checkpoints into the inference work directory cache and stores resolved local files in `artifacts["checkpoints"]`,

`registry list`:

- prints currently registered keys (including keys loaded from plugins).

`plugins list`:

- prints discovered Python entry points for plugin group `chimera_ml.plugins`.

Inference example from this repo:

```bash
pip install -e examples/oragen
chimera-ml inference -i video.mp4 -o out.json --config-path examples/oragen/configs/inference.yaml
```

For DAG inference configs:

- each step still uses the normal config shape: `name` plus optional `params`
- if `pipeline.parallel` is omitted or set to `false`, the pipeline runs sequentially in config order
- if `pipeline.parallel: true`, all steps become DAG nodes and `after` lists their explicit dependencies
- in parallel mode, steps without `after` become root nodes and may start immediately
- if parallel mode is enabled but no step has `after`, the builder emits a warning
- by default, dependency names refer to the step `name`
- use `id` only for the steps that need it, for example when the same step `name` is reused multiple times
- if two DAG steps write the same artifact key, the pipeline raises an error

Sequential example:

```yaml
steps:
  - name: resolve_checkpoints_step
    params:
      cache_dir: model_cache
      checkpoints:
        fusion: https://example.com/models/fusion.pt

  - name: extract_audio
    params:
      sample_rate: 16000
      mono: true

  - name: vad
    params:
      threshold: 0.5
      model: /path/to/vad

  - name: plugin_fusion_step
    params:
      checkpoint_key: fusion
```

Example where only repeated steps need `id`:

```yaml
pipeline:
  parallel: true

steps:
  - name: extract_audio
    params:
      sample_rate: 16000
      mono: true

  - name: sample_frames
    params:
      fps: 5

  - name: vad
    after: [extract_audio]
    params:
      threshold: 0.5
      model: /path/to/vad

  - id: detector_fast
    name: detect_faces
    after: [sample_frames]
    params:
      conf: 0.25
      model: /path/to/fast_face_detector

  - id: detector_accurate
    name: detect_faces
    after: [sample_frames]
    params:
      conf: 0.5
      model: /path/to/accurate_face_detector

  - name: build_windows
    after: [vad, detector_fast, detector_accurate]
    params:
      window_sec: 10
      stride_sec: 5
```

## YAML Config Model

Top-level sections used by runtime:

- `seed`
- `experiment_info` (required for `train`)
- `data`
- `model`
- `train`
- `loss`
- `optimizer`
- `scheduler` (optional)
- `metrics` (list)
- `logging` (list)
- `callbacks` (list)

Minimal skeleton:

```yaml
seed: 0

experiment_info:
  params:
    experiment_name: "my_experiment"
    include_time: true
    datetime_format: "%Y-%m-%d_%H-%M"
    timezone: "UTC"

data:
  name: "my_datamodule"
  params: {}

model:
  name: "my_model"
  params: {}

train:
  params:
    epochs: 10
    device: "cuda"
    mixed_precision: true
    use_scheduler: true
    scheduler_step_per_epoch: true
    scheduler_monitor: "val/loss"

loss:
  name: "mse_loss"
  params: {}

optimizer:
  name: "adamw_optimizer"
  params:
    lr: 1e-3

scheduler:
  name: "steplr_scheduler"
  params:
    step_size: 10
    gamma: 0.5

metrics:
  - name: "mae_metric"
    params: {}

logging:
  - name: "console_file_logger"
    params:
      log_path: "logs"
  - name: "mlflow_logger"
    params:
      tracking_uri: "sqlite:///logs/mlflow.db"

callbacks:
  - name: "checkpoint_callback"
    params:
      monitor: "val/loss"
      mode: "min"
```

Sweep grid example:

```yaml
parameters:
  optimizer.params.lr: [0.001, 0.0003, 0.0001]
  train.params.epochs: [3, 5]
```

Explicit trial example:

```yaml
trials:
  - optimizer.params.lr: 0.001
    train.params.epochs: 3
  - optimizer.params.lr: 0.0001
    callbacks.checkpoint_callback.params.monitor: "val/ccc"
```

## TrainConfig Parameters

`train.params` is mapped to `TrainConfig`:

- `epochs` (default `10`)
- `grad_clip_norm` (default `null`)
- `mixed_precision` (default `false`)
- `log_every_steps` (default `50`)
- `device` (default `"cuda"`)
- `train_loader_mode`: `single | round_robin | weighted` (default `single`)
- `train_stop_on`: `min | max` (default `min`)
- `train_loader_weights` (optional mapping for weighted mode)
- `use_scheduler` (default `false`)
- `scheduler_step_per_epoch` (default `true`)
- `scheduler_monitor` (optional metric key)
- `collect_cache` (default `true`)

## Data Contract

The trainer expects `Batch` objects:

```python
Batch(
    inputs={"modality": tensor, ...},
    targets=tensor_or_none,
    masks={"sequence_mask": ..., "audio_mask": ..., ...} or None,
    meta={"sample_meta": [...], ...} or None,
)
```

Built-in `MaskingCollate` (`masking_collate`) supports variable-length multimodal inputs and creates:

- padded modality tensors,
- `sequence_mask`,
- per-modality masks like `{modality}_mask`,
- optional legacy `meta["masks"]`.

## Plugin System

On `import chimera_ml`, the library calls `register_all()`:

- loads built-in modules that register built-in components,
- loads external entry points from group `chimera_ml.plugins`,
- executes callable entry points once per process.

Plugin declaration (recommended, PEP 621 style):

```toml
[project.entry-points."chimera_ml.plugins"]
my_project = "my_project.chimera_plugin:register"
```

## BuildContext For Plugin Authors

During `train` and `eval`, `chimera-ml` creates a per-run `BuildContext` and passes it through the build pipeline:

- `datamodule`
- `model`
- `loss`
- `metrics`
- `optimizer`
- `scheduler`
- `callbacks`
- `collate`
- `logger`

Use it when downstream components need runtime metadata that should not be duplicated in YAML, such as class names, number of classes, class weights, window sizes, output schema, or metric names. `BuildContext` is intended for shared metadata, not for passing live runtime objects between components.

Factories can accept an optional `context` argument:

```python
from chimera_ml.core.registry import LOSSES


@LOSSES.register("my_loss")
def my_loss(alpha: float = 1.0, context = None):
    class_weights = context.get("data.class_weights") if context is not None else None
    return MyLoss(alpha=alpha, class_weights=class_weights)
```

Built components can also enrich the context during registration by implementing `describe_context(...)`:

```python
class MyDataModule(DataModule):
    def describe_context(self, context) -> None:
        context.set("data.num_classes", 3)
        context.set("data.class_names", ["negative", "neutral", "positive"])
        context.set("data.class_weights", [0.2, 0.5, 0.3])
```

`BuildContext` is local to a single CLI run. It is not a global singleton, so it remains safe for tests, sweeps, and independent experiments. Runtime metadata such as the current config and stage are available directly as `context.config` and `context.stage`.

Typical `register()` function:

```python
def register():
    import my_project.data
    import my_project.models
    import my_project.losses
    import my_project.metrics
    import my_project.callbacks
```

Import side effects execute registry decorators.

## Built-In Registry Keys

Datamodules are intentionally project-specific. Built-in `DATAMODULES` is empty by default.

`MODELS`:

- `feature_fusion_model`
- `prediction_fusion_model`
- `gated_fusion_model`
- `gated_prediction_fusion_model`

`LOSSES`:

- `mse_loss`
- `mae_loss`
- `cross_entropy_loss`
- `focal_loss`
- `bce_with_logits_loss`
- `ccc_loss`

`METRICS`:

- `mae_metric`
- `mse_metric`
- `rmse_metric`
- `r2_metric`
- `prf_macro_metric`
- `prf_micro_metric`
- `prf_weighted_metric`
- `confusion_matrix_metric`

`OPTIMIZERS`:

- `adamw_optimizer`
- `adam_optimizer`
- `sgd_optimizer`

`SCHEDULERS`:

- `steplr_scheduler`
- `cosineannealinglr_scheduler`
- `reduceonplateau_scheduler`

`CALLBACKS`:

- `checkpoint_callback`
- `collect_predictions_callback`
- `early_stopping_callback`
- `snapshot_callback`
- `telegram_notifier_callback`

`LOGGERS`:

- `console_file_logger`
- `mlflow_logger`

`COLLATES`:

- `masking_collate`

## Training and Evaluation Behavior

- Mixed precision uses `torch.amp.autocast` and `GradScaler` on CUDA.
- Training supports one or many train loaders.
- Multi-loader scheduling modes:
  - `single`: first loader only.
  - `round_robin`: cycle loaders.
  - `weighted`: stochastic sampling with loader weights.
- Validation/test loaders are normalized to stable split keys (`normalize_loaders`).
- Metrics are stateful (`reset -> update -> compute`).
- Optional prediction cache (`CachedSplitOutputs`) stores CPU preds/targets/features for callbacks.

## Logging and Artifacts

`console_file_logger`:

- logs to console + file under `<log_path>/<experiment_name>/<run_name>/`.

`mlflow_logger`:

- logs params/metrics,
- supports file/text/bytes artifact logging,
- logs config artifact when `config_path` is provided.

`plot_confusion_matrix_callback`:

- builds confusion matrix figures from cached predictions,
- logs PNG artifacts to MLflow per split (`figures/<split>/...`).

`telegram_notifier_callback`:

- sends final run status to Telegram via Bot API.

## Callback Lifecycle

Callbacks follow:

- `on_fit_start`
- `on_epoch_start`
- `on_batch_end`
- `on_epoch_end`
- `on_fit_end`

Highlights:

- `checkpoint_callback`: monitor-based top-k and `last.pt`.
- `early_stopping_callback`: monitor, mode, patience, min_delta.
- `snapshot_callback`: code/config snapshots (`code.zip`, config copy).
- `collect_predictions_callback`: CSV prediction artifacts to MLflow.
- `plot_confusion_matrix_callback`: confusion matrix PNG artifacts to MLflow.
- `telegram_notifier_callback`: final Telegram notification via env vars.

## Repository Example

`examples/va_estimation` is a full plugin package using entry points and task-specific components. Use it as a template for new projects.

## Development

Quality checks:

```bash
poetry run ruff check src tests
poetry run pytest
```

See `CONTRIBUTING.md` for contribution details.

## Publishing

PyPI publishing is automated via GitHub Actions workflow:

- `.github/workflows/publish.yml` (triggered by GitHub Release `published`).

Release checklist:

- see `RELEASING.md`.
