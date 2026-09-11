# Sweeps

chimera-ml sweep runs normal training once per generated trial:

~~~bash
chimera-ml sweep \
  --base-config config.yaml \
  --sweep-config sweep.yaml \
  --sweep-name lr-search \
  --max-trials 4 \
  --dry-run
~~~

The base config must contain experiment_info.params.experiment_name. Overrides
use dotted paths and can address named list entries, for example
optimizer.params.lr or callbacks.checkpoint_callback.params.monitor.

## Grid sweeps

Omit method or use method: grid. Define either parameters or an explicit
trials list, never both:

~~~yaml
parameters:
  optimizer.params.lr: [0.001, 0.0001]
  train.params.epochs: [3, 5]
~~~

The Cartesian product creates four trials. An explicit list is useful for
hand-selected combinations.

## Optuna sweeps

Set method: optuna and describe typed spaces:

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

Float supports low, high, optional step or log; int supports low, high, step,
and log; categorical requires non-empty choices. step and log cannot be
combined for floats; integer log search requires step 1. storage and
load_if_exists are also accepted.

The CLI adds sweep_target_callback to each Optuna trial. It tracks the best
scalar monitor; the study minimizes or maximizes according to target.mode.
The target defaults to val/loss and min.

## Artifacts and identity

~~~text
<log_path>/<experiment_name>/_sweeps/<sweep_id>/
├── base_config.yaml
├── sweep_config.yaml
├── manifest.yaml
└── trial_configs/<label>-<short-id>-001.yaml
~~~

log_path comes from the first grid trial's console logger path, or the
configured logger root. A human name is included in trial ids when
--sweep-name is supplied. manifest.yaml records status, trial records, and for
Optuna the study/objective/best trial.

--max-trials limits grid trials and caps Optuna n_trials; --dry-run prints
generated trial information without starting training.
