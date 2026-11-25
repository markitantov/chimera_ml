# Sweep objective callback — `SweepTargetCallback`

`SweepTargetCallback`, registry key `sweep_target_callback`, tracks the best scalar value for an Optuna/grid sweep. Parameters are `monitor="val/loss"` and `mode="min"` or `"max"`. It resets at `on_fit_start`, records available keys and the last value at `on_epoch_end`, and stores `best_value`, `best_epoch`, `last_value`, `last_epoch`, and `available_keys`. It does not stop training or write files.

The sweep workflow normally inserts this callback automatically. Manual use is only needed when a caller wants the tracked state:

```yaml
- name: sweep_target_callback
  params: {monitor: val/mae, mode: min}
```

The monitor must be a scalar split-prefixed log key. Missing keys are recorded in `available_keys` and do not update the objective.
