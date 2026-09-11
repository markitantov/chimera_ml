# Sweeps

chimera-ml sweep запускает обычный training для каждого trial:

~~~bash
chimera-ml sweep \
  --base-config config.yaml \
  --sweep-config sweep.yaml \
  --sweep-name lr-search \
  --max-trials 4 \
  --dry-run
~~~

Base config должен содержать experiment_info.params.experiment_name. Overrides
используют dotted paths, включая named list entries:
optimizer.params.lr или callbacks.checkpoint_callback.params.monitor.

## Grid sweeps

Без method (или method: grid) задайте parameters либо trials, но не оба:

~~~yaml
parameters:
  optimizer.params.lr: [0.001, 0.0001]
  train.params.epochs: [3, 5]
~~~

Cartesian product создаёт четыре trials. trials удобен для ручных комбинаций.

## Optuna sweeps

~~~yaml
method: optuna
n_trials: 10
study_name: lr-study
target:
  monitor: val/loss
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
~~~

float требует low/high, допускает step или log; int допускает low/high/step/log;
categorical требует непустой choices. storage и load_if_exists также
поддерживаются. CLI добавляет sweep_target_callback, target по умолчанию
val/loss и min.

## Artifacts и identity

~~~text
<log_path>/<experiment_name>/_sweeps/<sweep_id>/
├── base_config.yaml
├── sweep_config.yaml
├── manifest.yaml
└── trial_configs/<label>-<short-id>-001.yaml
~~~

manifest фиксирует status, trials и для Optuna study/objective/best trial.
--max-trials ограничивает trials; --dry-run печатает их и не запускает training.
