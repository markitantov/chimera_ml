# Hyperparameter sweeps

Sweep берёт valid training config и запускает обычный training для каждого trial. Сначала проверьте base config; sweep не исправляет datasets, plugin keys или missing monitors.

## Grid workflow

`sweep.yaml` содержит либо `parameters`, либо `trials`, но не оба:

```yaml
parameters:
  optimizer.params.lr: [0.001, 0.0001]
  train.params.epochs: [3, 5]
```

Это даёт четыре Cartesian trials:

```bash
chimera-ml validate-config --config-path config.yaml
chimera-ml sweep --base-config config.yaml --sweep-config sweep.yaml --sweep-name lr-search --max-trials 4 --dry-run
chimera-ml sweep --base-config config.yaml --sweep-config sweep.yaml --sweep-name lr-search
```

Dotted paths могут адресовать mappings и named list entries: `optimizer.params.lr`, `callbacks.checkpoint_callback.params.monitor`. `--max-trials` ограничивает grid, `--dry-run` не запускает training.

## Optuna workflow

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

Float требует `low`/`high`, использует `step` или `log`, но не оба. Int имеет `step` default `1`; log для int требует `step: 1`. Categorical требует non-empty `choices`. `target.mode` — `min`/`max` и aliases `minimize`/`maximize`; default objective — `val/loss`, `min`.

```bash
chimera-ml sweep -b config.yaml -s optuna.yaml --dry-run
chimera-ml sweep -b config.yaml -s optuna.yaml --max-trials 3
```

CLI добавляет `sweep_target_callback`, запускает normal train flow и записывает best value/epoch.

## Artifacts

```text
<log_path>/<experiment_name>/_sweeps/<sweep-id>/
├── base_config.yaml
├── sweep_config.yaml
├── manifest.yaml
└── trial_configs/<label>-<short-id>-001.yaml
```

`manifest.yaml` содержит status и trial records; Optuna дополнительно сохраняет study/objective/best trial. Каждый trial получает own run artifacts.
