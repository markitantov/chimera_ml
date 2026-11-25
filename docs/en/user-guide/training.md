# Running training

## Validate, then train

Use a complete, plugin-backed config:

```bash
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml train --config-path examples/va_estimation/configs/multimodal_train.yaml
```

`validate-config` parses YAML and checks required framework sections without constructing datasets or starting training. `train` loads built-in and entry-point plugins, seeds the process, requires `experiment_info.params.experiment_name`, generates a run name, builds the DataModule/model/loss/metrics/optimizer/scheduler/callbacks/loggers, and calls `Trainer.fit`.

## What `Trainer.fit` does

For each epoch, the Trainer:

1. invokes `on_fit_start` once and `on_epoch_start` for callbacks;
2. iterates training batches, moves them to the resolved device, calls the model, computes the loss, checks finite values, backpropagates, optionally clips gradients, steps the optimizer, and updates metrics;
3. optionally steps a batch scheduler after every optimizer update;
4. runs each validation loader with gradients disabled and updates validation metrics/cache;
5. logs train/validation metrics and learning rates, invokes `on_epoch_end`, then steps an epoch scheduler when configured;
6. stops early if a callback sets `trainer.stop_training`, and finally invokes `on_fit_end`.

Training requires `batch.targets`. An unlabeled batch raises a clear error; use evaluation or an inference step for unlabeled data.

## Devices, precision, and clipping

`device: cuda` uses CUDA when available and otherwise resolves to CPU. `mixed_precision: true` enables `torch.amp.autocast` and a gradient scaler only on CUDA. `grad_clip_norm: 1.0` unscales the optimizer before applying `clip_grad_norm_`; `null` disables clipping.

```yaml
train:
  params:
    epochs: 10
    device: cuda
    mixed_precision: true
    grad_clip_norm: 1.0
```

## Multiple loaders

A DataModule can return a mapping or sequence of training loaders. Choose how batches are sampled:

```yaml
train:
  params:
    train_loader_mode: round_robin
    train_stop_on: min
```

`single` is the default. `round_robin` alternates loaders; `weighted` samples according to `train_loader_weights`. `train_stop_on: min` ends a mixed epoch when the first loader is exhausted; `max` continues until the last loader is exhausted. For weighted mode, provide positive weights keyed by normalized loader name.

## Optimizer and scheduler stepping

The built-in optimizer keys are `adamw_optimizer`, `adam_optimizer`, and `sgd_optimizer`. Scheduler keys are `steplr_scheduler`, `cosineannealinglr_scheduler`, and `reduceonplateau_scheduler`; their `params` are passed to the corresponding PyTorch scheduler.

```yaml
optimizer:
  name: adamw_optimizer
  params:
    lr: 0.001
    weight_decay: 0.01
scheduler:
  name: reduceonplateau_scheduler
  params:
    factor: 0.5
    patience: 2
train:
  params:
    use_scheduler: true
    scheduler_step_per_epoch: true
    scheduler_monitor: val/loss
```

With `scheduler_step_per_epoch: false`, the scheduler steps after each optimizer update. With epoch stepping, metric-aware schedulers receive the flat callback log mapping and use `scheduler_monitor` when supplied. A scheduler section alone does not enable stepping; set `use_scheduler: true`.

## Metrics and callbacks

Epoch logs are flat keys. Trainer prefixes metric mappings with the split name, so a metric returning `mae` becomes `train/mae` or `val/mae`; plugin callbacks may add keys such as `val_framewise/va_ccc_metric`. Use the exact emitted key for `monitor` in checkpoint, early stopping, scheduler, and sweep callbacks.

Callbacks run at lifecycle events and can save checkpoints, cache-dependent predictions, snapshots, figures, or notifications. Callback ordering matters when a callback consumes cached validation outputs; callbacks that need predictions set `collect_cache: true` themselves, but setting it explicitly makes the dependency clear.

## Outputs

With the example's `log_path: logs` and experiment name `va_estimation_fusion`:

```text
logs/<experiment>/<run>/
├── train.log
├── checkpoints/
│   ├── last.pt
│   └── epoch=...pt
├── config.yaml       # if snapshot_callback.save_config is true
└── code.zip          # if snapshot_callback.save_code_zip is true
```

A checkpoint payload contains `epoch`, `global_step`, `model_state_dict`, `optimizer_state_dict`, and `scheduler_state_dict` when a scheduler exists. The checkpoint callback saves `last.pt` when `save_last: true` and retains monitored improvements according to `mode` and `save_top_k`.

## Common training errors

- **Missing `experiment_info.params.experiment_name`**: add it for `train` and `sweep`.
- **Unknown registry key**: install the plugin and check `chimera-ml plugins list` and `chimera-ml registry list --type ...`.
- **No training targets**: fix the dataset/collate so `Batch.targets` is populated.
- **Monitor not found**: inspect the validation logs and use the exact prefixed metric key.
- **Shape or non-finite error**: compare modality/target shapes with the model and loss contract; do not disable masks to hide incompatible data.
