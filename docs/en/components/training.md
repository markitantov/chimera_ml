# Training components

## Loss contract and registry

A `BaseLoss` implements `__call__(output: ModelOutput, batch: Batch) -> torch.Tensor`. The returned differentiable tensor is consumed by the Trainer. Built-in keys are:

| Key | Expected data | Parameters/defaults |
| --- | --- | --- |
| `mse_loss` | Same-shaped regression tensors | `reduction="mean"` (`mean`, `sum`, `none`) |
| `mae_loss` | Same-shaped regression tensors | Same reduction options |
| `ccc_loss` | Same-shaped regression tensors, flattened to `[B,D]` | `eps=1e-12`; returns `1 - mean(CCC)` |
| `cross_entropy_loss` | Logits `[N,C]` and integer class targets | `label_smoothing=0.0` |
| `focal_loss` | Multi-class logits and integer targets | `gamma=2.0`, `alpha=None`, `reduction="mean"`, `label_smoothing=0.0` |
| `bce_with_logits_loss` | Multi-label logits/float targets | `pos_weight=None`, `reduction="mean"` |

No built-in loss applies temporal masks, ignores missing labels, or consumes `ModelOutput.aux` automatically. Shapes and target semantics must match the selected loss.

```yaml
loss:
  name: cross_entropy_loss
  params: {label_smoothing: 0.1}
```

## Metrics

`BaseMetric` is stateful: Trainer calls `reset()` at split/epoch start, `update(output, batch)` for labeled batches, and `compute()` at epoch end. Built-ins are:

| Keys | Output | Parameters |
| --- | --- | --- |
| `mae_metric`, `mse_metric`, `rmse_metric` | `mae`, `mse`, `rmse` | `multioutput`: `uniform_average` default or `raw_values` |
| `r2_metric` | `r2` | Also `variance_weighted`; one-sample data yields NaN values |
| `prf_micro_metric`, `prf_macro_metric`, `prf_weighted_metric` | `<average>_precision`, `<average>_recall`, `<average>_f1` | `zero_division=0` |
| `confusion_matrix_metric` | `cm_acc`; matrix via `.value()` | `normalize=None`, `true`, `pred`, or `all` |

Regression metrics flatten each output to `[N,D]` and require matching shapes. PRF/confusion metrics use `argmax(-1)` class predictions and flattened integer targets. Empty state returns `{}`. Trainer prefixes split logs, for example `train/mae`, `val/r2`, or `val/macro_f1`; callbacks, scheduler monitors, and sweep objectives must use those exact keys.

```yaml
metrics:
  - name: mae_metric
    params: {multioutput: uniform_average}
  - name: prf_macro_metric
    params: {zero_division: 0}
```

## Optimizers and schedulers

Factories inject `model` or `optimizer`; remaining parameters pass to PyTorch:

| Key | Underlying class | Factory defaults |
| --- | --- | --- |
| `adamw_optimizer` | `torch.optim.AdamW` | `lr=1e-3`, `weight_decay=0.0` |
| `adam_optimizer` | `torch.optim.Adam` | same |
| `sgd_optimizer` | `torch.optim.SGD` | `lr=1e-2`, `weight_decay=0.0`, `momentum=0.0`, `nesterov=False` |

Schedulers are `steplr_scheduler` -> `StepLR`, `cosineannealinglr_scheduler` -> `CosineAnnealingLR`, and `reduceonplateau_scheduler` -> `ReduceLROnPlateau`. Their parameters are passed directly to PyTorch; `optimizer` is injected. Required parameters therefore follow the underlying class, such as `step_size`/`gamma`, `T_max`, or `mode`/`patience`.

```yaml
optimizer:
  name: adamw_optimizer
  params: {lr: 0.001, weight_decay: 0.01}
scheduler:
  name: cosineannealinglr_scheduler
  params: {T_max: 50}
train:
  params:
    use_scheduler: true
    scheduler_step_per_epoch: true
```

`use_scheduler` is Trainer configuration; it determines whether a configured scheduler is stepped. `scheduler_step_per_epoch` selects epoch-level versus batch-level stepping. `scheduler_monitor` is the Trainer monitor used for metric-based schedulers such as `ReduceLROnPlateau`; it must match the split-prefixed log key. These are not PyTorch scheduler constructor arguments.
