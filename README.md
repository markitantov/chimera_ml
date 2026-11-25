# Chimera ML

`Chimera ML (Cross-modal Hierarchical Merging of Embeddings and Representations)` is a lightweight framework for configurable unimodal and multimodal training, checkpoint evaluation, and registry-driven inference pipelines.

The framework provides infrastructure: a Trainer, logging, callbacks, registry builders, sequential or parallel DAG execution, modality-intervention XAI, and four built-in feature-only fusion architectures. Task-specific datasets, models, losses, metrics, prediction steps, and other components usually live in external plugin packages. YAML configuration and CLI commands connect that infrastructure to an experiment without making task-specific assumptions part of the core library.

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

The core package does not provide task-specific datasets or prediction models. Install a compatible plugin, prepare the external datasets and cached features it requires, and obtain the model weights or checkpoints required by that plugin before training or inference.

Example plugin from this repo:

```bash
pip install -e examples/va_estimation
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml train --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml eval --config-path examples/va_estimation/configs/multimodal_test.yaml --checkpoint-path path/to/last.pt
```

## CLI

The following is schematic CLI syntax; replace angle-bracket placeholders with actual paths and values.

Main commands:

```text
chimera-ml validate-config --config-path <config.yaml>
chimera-ml doctor
chimera-ml train --config-path <config.yaml>
chimera-ml sweep --base-config <config.yaml> --sweep-config <sweep.yaml> [--sweep-name NAME] [--max-trials N] [--dry-run]
chimera-ml eval --config-path <config.yaml> --checkpoint-path <ckpt.pt> [--with-features]
chimera-ml inference -i <input.mp4> [-o <out.json>] --config-path <inference.yaml> [--device cpu|cuda|auto] [--work-dir <dir>]
chimera-ml registry list [--type models|losses|metrics|optimizers|schedulers|callbacks|collates|loggers|datamodules|inference_steps]
chimera-ml plugins list [--group chimera_ml.plugins]
```

`validate-config`:

- checks training/evaluation config structure and required sections without starting training; it does not validate an inference YAML.

`doctor`:

- prints quick environment diagnostics (Python, torch, CUDA, MLflow, registry/plugin counts).

`train`:

- requires `experiment_info.params.experiment_name`,
- generates `run_name` via `generate_run_name(...)`,
- patches `checkpoint_callback` and `snapshot_callback` params with experiment/run data,
- builds all components from registries and runs `Trainer.fit(...)`.

`sweep`:

- supports Cartesian grids via `parameters` (grid search) and explicit trial lists via `trials`,
- supports Optuna sweeps via `method: optuna` and typed search spaces,
- keeps the existing grid/trial behavior when `method` is omitted,
- applies overrides using dotted paths such as `optimizer.params.lr` or
  `callbacks.checkpoint_callback.params.monitor`,
- stores sweep metadata under `<log_path>/<experiment_name>/_sweeps/<sweep_id>/`,
- saves `base_config.yaml`, `sweep_config.yaml`, `manifest.yaml`, and per-trial configs for each sweep series,
- runs the normal `train` flow once per trial and appends trial ids such as
  `lr-search-a1b2-001` to run names,
- supports `--max-trials` for CI smoke tuning and `--dry-run` to inspect generated trials.

`eval`:

- builds datamodule/model/loss/metrics/callbacks from the same config style,
- requires `--checkpoint-path` and loads either `model_state_dict` or a raw state dict from that checkpoint,
- evaluates over merged `train`/`val`/`test` loader splits when available.

`inference`:

- loads a small inference pipeline from YAML,
- builds steps from the `inference_steps` registry,
- runs them on a shared `InferenceContext`,
- runs sequentially by default in plain config order,
- enables DAG/parallel scheduling only when `pipeline.parallel: true` is set,
- uses `after` for explicit dependencies between steps in parallel mode,
- lets `--output/-o` create or override `write_json_predictions_step` only for sequential configs,
- requires `write_json_predictions_step` to be declared explicitly in the config when `pipeline.parallel: true`, together with its `after` dependencies,
- keeps output behavior inside explicit pipeline steps such as `write_json_predictions_step` and `print_json_predictions_step`,
- supports a built-in `resolve_checkpoints_step` whose default `cache_dir` is `checkpoints` relative to the inference work directory; it resolves local paths or downloads remote checkpoints and stores resolved local files in `artifacts["checkpoints"]`.

Core inference provides checkpoint resolution and JSON output steps; task-specific extraction, feature, fusion, and prediction steps are supplied by plugins.

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
- shared upstream artifacts may be read by multiple branches; overwrite conflicts are checked only for keys explicitly written via `InferenceContext.set_artifact(...)`
- writes to the same artifact key are rejected when they come from unordered parallel branches; an explicit dependency order permits the later step to overwrite the earlier value

The following YAML is schematic; replace placeholder paths and registry keys before use.

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

The following YAML is also schematic and demonstrates only the `id` rule.

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

## Explainable AI (XAI)

Chimera ML provides leave-one-modality-out sensitivity analysis through the registered `xai_step`. It reuses the original model prediction as the baseline and applies one intervention per selected modality using `zero`, `mask`, or `remove`. A wrapped prediction step may expose the optional `on_prediction` callback; XAI temporarily collects that callback and restores the previous value afterward.

The resulting artifacts support per-sample, global, and temporal summaries. This is intervention sensitivity, not causal attribution, SHAP, calibrated confidence, or a reliability guarantee. Existing ORAGEN and Affective States prediction steps do not expose the required callback and therefore do not support XAI as currently documented. See the full [English XAI guide](https://chimera-ml.readthedocs.io/en/latest/user-guide/xai/) and [Russian XAI guide](https://chimera-ml.readthedocs.io/ru/latest/user-guide/xai/).

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

Optuna example:

```yaml
method: optuna
n_trials: 30
objective:
  monitor: val/loss
  mode: min
parameters:
  optimizer.params.lr:
    type: float
    low: 1.0e-5
    high: 1.0e-2
    log: true
  optimizer.params.weight_decay:
    type: float
    low: 0.0
    high: 0.1
  train.params.epochs:
    type: int
    low: 3
    high: 10
  model.params.hidden_dim:
    type: categorical
    choices: [128, 256, 512]
```

Sweep execution:

- When `method` is omitted, empty, or `cartesian`, `chimera-ml sweep` uses `GridSweep`.
- Grid sweeps read either `parameters` as a Cartesian product or `trials` as explicit override mappings.
- `method: optuna` uses `OptunaSweep`; `parameters` becomes an Optuna search space with `float`, `int`, and `categorical` values.
- Optuna target config can be written as `objective`, `target`, or `metric`; it defaults to `monitor: val/loss`, `mode: min`.
- For Optuna trials, the CLI materializes the trial YAML, appends `sweep_target_callback` to `callbacks`, runs the normal train flow, and returns the callback's best value to `study.optimize(...)`.
- `--dry-run` prints generated grid trials or the Optuna search space without creating sweep artifacts or starting training.

Sweep artifacts:

```text
<log_path>/<experiment_name>/_sweeps/<sweep_id>/
  base_config.yaml
  sweep_config.yaml
  manifest.yaml
  trial_configs/
    <sweep_name>-<short_id>-001.yaml
    <sweep_name>-<short_id>-002.yaml
```

`<log_path>` comes from `logging.console_file_logger.params.log_path` and defaults to `logs`. Grid sweeps resolve it after applying the first trial override, so a grid override of the console logger path moves the whole sweep folder.

For grid sweeps, `manifest.yaml` records the sweep identity, status, timestamps, and `runs` entries with `trial_id` and generated `run_name`. For Optuna sweeps it also records `method`, `study_name`, `objective`, `n_trials`, per-trial objective `value`, optional `target_epoch`, sampled `overrides`, and `best_trial`.

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

Built-in collates support two data contracts:

- MaskingCollate (masking_collate) preserves the legacy padded inputs,
  sequence_mask, per-modality presence masks, and optional legacy meta masks.
- MultimodalSequenceCollate (multimodal_sequence_collate) pads each temporal
  modality independently and emits float32 `<modality>_mask` presence masks
  plus boolean `<modality>_temporal_mask` masks. It supports missing and zero-length
  sequences, explicit temporal masks, pad_sequences, and pad_value.

The new collate does not align timelines automatically and does not replace
MaskingCollate. Fusion utilities prefer dedicated temporal masks and validate
them against normalized feature shapes.

During CLI `train` and `eval`, an explicitly present top-level `collate:`
section is resolved with `build_collate()` and passed to the datamodule factory
as `collate_fn`. A custom datamodule must declare a keyword-capable `collate_fn`
parameter; datamodules without that contract fail with an actionable error.
When `collate:` is absent, the CLI preserves the existing datamodule construction
and does not inject an extra collate. Do not configure both top-level `collate:`
and `data.params.collate_fn`.

## Built-In Fusion Models

The built-in fusion models consume precomputed feature tensors from
Batch.inputs. They create input projections and prediction layers internally;
constructors do not receive external encoder dictionaries or prediction heads.

| Model | Use it for | Prediction |
| --- | --- | --- |
| tfs_fusion_model | MASAI-style directed hierarchical fusion | [B, output_dim] |
| tfs_dcmmmoe_fusion_model | TFS with adaptive masked pair gates | [B, output_dim] |
| raav_fusion_model | Aligned frame streams with context latents | [B, T, output_dim] |
| tomar_fusion_model | Anchor-conditioned gated residual fusion | [B, output_dim] |

Python construction:

~~~python
from chimera_ml.models import TFSFusionModel

model = TFSFusionModel(
    input_dims={"text": 1024, "audio": 768, "video": 512},
    shared_dim=256,
    output_dim=2,
)
~~~

YAML construction:

~~~yaml
model:
  name: tfs_fusion_model
  params:
    input_dims: {text: 1024, audio: 768, video: 512}
    shared_dim: 256
    output_dim: 2

collate:
  name: multimodal_sequence_collate
  params:
    pad_sequences: true
~~~

See the detailed English fusion guides at docs/en/user-guide/models.md and
Russian fusion guides at docs/ru/user-guide/models.md.

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

- `tfs_fusion_model`
- `tfs_dcmmmoe_fusion_model`
- `raav_fusion_model`
- `tomar_fusion_model`

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
- `multimodal_sequence_collate`

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
- logs PDF artifacts to MLflow per split (`figures/<split>/...`).

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
- `plot_confusion_matrix_callback`: confusion matrix PDF artifacts to MLflow.
- `telegram_notifier_callback`: final Telegram notification via env vars.

## Repository Example

`examples/va_estimation` is a full plugin package using entry points and task-specific components. Use it as a template for new projects.

`examples/oragen` is a plugin package for audio-visual gender recognition and age estimation.

`examples/affective_states_recognition` is a multimodal plugin package for multimodal and multi-task emotion and sentiment recognition over audio, video, and text, with train/eval/inference configs and accompanying docs.

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

## Documentation

The English documentation website is published on [Read the Docs](https://chimera-ml.readthedocs.io/). The Russian documentation is maintained in `docs/ru/` and built with `mkdocs.ru.yml`:

- [English documentation](https://chimera-ml.readthedocs.io/en/latest/)
- [English getting started](https://chimera-ml.readthedocs.io/en/latest/getting-started/)
- [English training and evaluation](https://chimera-ml.readthedocs.io/en/latest/user-guide/training/)
- [English inference and XAI](https://chimera-ml.readthedocs.io/en/latest/user-guide/inference/)
- [Russian documentation](https://github.com/markitantov/chimera_ml/tree/main/docs/ru)
- [Russian начало работы](https://github.com/markitantov/chimera_ml/tree/main/docs/ru/getting-started)
- [Russian обучение и оценка](https://github.com/markitantov/chimera_ml/blob/main/docs/ru/user-guide/training.md)
- [Russian инференс и XAI](https://github.com/markitantov/chimera_ml/blob/main/docs/ru/user-guide/inference.md)
