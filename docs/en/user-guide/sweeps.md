# Hyperparameter sweeps

A sweep starts from one valid training configuration and invokes the normal training flow for each trial. Prepare and validate the base config first; a sweep does not make missing datasets, plugin keys, or monitored metrics valid.

## Grid workflow

Create `sweep.yaml` with either a `parameters` mapping or an explicit `trials` list, never both:

```yaml
parameters:
  optimizer.params.lr: [0.001, 0.0001]
  train.params.epochs: [3, 5]
```

This produces four Cartesian trials. Run a dry check, then launch:

```bash
chimera-ml validate-config --config-path config.yaml
chimera-ml sweep   --base-config config.yaml   --sweep-config sweep.yaml   --sweep-name lr-search   --max-trials 4   --dry-run
chimera-ml sweep   --base-config config.yaml   --sweep-config sweep.yaml   --sweep-name lr-search
```

Dotted paths address mappings and named list entries, for example `optimizer.params.lr` or `callbacks.checkpoint_callback.params.monitor`. `--max-trials` truncates the generated grid. `--dry-run` prints trial overrides and does not start training.

## Optuna workflow

Use typed search spaces and one scalar objective:

```yaml
method: optuna
n_trials: 10
study_name: lr-study
target:
  monitor: val/mae
  mode: min
parameters:
  optimizer.params.lr:
    type: float
    low: 0.00001
    high: 0.001
    log: true
  train.params.epochs:
    type: int
    low: 3
    high: 20
    step: 1
  model.params.variant:
    type: categorical
    choices: [small, large]
```

Float spaces require `low` and `high`; they may use `step` or `log`, but not both. Integer spaces use `low`, `high`, optional `step` (default `1`), and optional `log`; logarithmic integers require `step: 1`. Categorical spaces require a non-empty `choices` list. `target.mode` is `min` or `max` and also accepts `minimize`/`maximize` aliases. If omitted, the objective defaults to `val/loss` and `min`.

```bash
chimera-ml sweep -b config.yaml -s optuna.yaml --dry-run
chimera-ml sweep -b config.yaml -s optuna.yaml --max-trials 3
```

The CLI adds `sweep_target_callback` to each Optuna trial, runs the normal `train` flow, and records the best value and epoch. Optuna must be installed; it is a core runtime dependency in this project, but the command still fails clearly if the environment lacks it.

## Artifacts and results

Sweep metadata uses the base/configured console logger's `log_path`:

```text
<log_path>/<experiment_name>/_sweeps/<sweep-id>/
├── base_config.yaml
├── sweep_config.yaml
├── manifest.yaml
└── trial_configs/<label>-<short-id>-001.yaml
```

`manifest.yaml` records status and trial records; Optuna additionally records study/objective/best-trial information. Each trial has its own generated run name and ordinary training artifacts.
