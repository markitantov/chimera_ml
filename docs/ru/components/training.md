# Training components

## Losses

`BaseLoss.__call__(output: ModelOutput, batch: Batch) -> torch.Tensor` возвращает differentiable tensor для Trainer.

| Key | Данные | Parameters/defaults |
| --- | --- | --- |
| `mse_loss` | Same-shaped regression | `reduction="mean"` |
| `mae_loss` | Same-shaped regression | `reduction="mean"` |
| `ccc_loss` | Same-shaped regression, flatten в `[B,D]` | `eps=1e-12`, loss `1 - mean(CCC)` |
| `cross_entropy_loss` | logits `[N,C]`, integer targets | `label_smoothing=0.0` |
| `focal_loss` | multiclass logits и integer targets | `gamma=2.0`, `alpha=None`, `reduction="mean"`, `label_smoothing=0.0` |
| `bce_with_logits_loss` | multilabel logits/float targets | `pos_weight=None`, `reduction="mean"` |

Built-in losses не применяют temporal masks, missing-label handling или `ModelOutput.aux` автоматически.

```yaml
loss:
  name: focal_loss
  params: {gamma: 2.0, reduction: mean}
```

## Metrics

`BaseMetric` stateful: Trainer вызывает `reset()`, затем `update(output, batch)` для labeled batches и `compute()` в конце epoch.

| Keys | Output | Parameters |
| --- | --- | --- |
| `mae_metric`, `mse_metric`, `rmse_metric` | `mae`, `mse`, `rmse` | `multioutput`: `uniform_average` default или `raw_values` |
| `r2_metric` | `r2` | Также `variance_weighted`; one-sample может дать NaN |
| `prf_micro_metric`, `prf_macro_metric`, `prf_weighted_metric` | `<average>_precision`, `<average>_recall`, `<average>_f1` | `zero_division=0` |
| `confusion_matrix_metric` | `cm_acc`, matrix через `.value()` | `normalize=None`, `true`, `pred`, `all` |

Regression flatten в `[N,D]` и требует matching shapes. PRF/confusion используют `argmax(-1)` и flattened integer targets. Trainer добавляет split prefix: `train/mae`, `val/r2`, `val/macro_f1`. Именно эти keys используются callbacks/scheduler/sweep.

```yaml
metrics:
  - name: mae_metric
    params: {multioutput: uniform_average}
  - name: prf_macro_metric
    params: {zero_division: 0}
```

## Optimizers и schedulers

| Key | PyTorch | Factory defaults |
| --- | --- | --- |
| `adamw_optimizer` | `torch.optim.AdamW` | `lr=1e-3`, `weight_decay=0.0` |
| `adam_optimizer` | `torch.optim.Adam` | те же |
| `sgd_optimizer` | `torch.optim.SGD` | `lr=1e-2`, `weight_decay=0.0`, `momentum=0.0`, `nesterov=False` |

Schedulers: `steplr_scheduler` → `StepLR`, `cosineannealinglr_scheduler` → `CosineAnnealingLR`, `reduceonplateau_scheduler` → `ReduceLROnPlateau`; их params передаются underlying PyTorch class, `optimizer` injects builder.

```yaml
optimizer:
  name: adamw_optimizer
  params: {lr: 0.001, weight_decay: 0.01}
scheduler:
  name: cosineannealinglr_scheduler
  params: {T_max: 50}
train:
  params: {use_scheduler: true, scheduler_step_per_epoch: true}
```

`use_scheduler`, `scheduler_step_per_epoch` и `scheduler_monitor` — Trainer settings, не PyTorch constructor args. Первый включает stepping, второй выбирает batch/epoch stepping, третий задаёт split-prefixed metric для `ReduceLROnPlateau`.
