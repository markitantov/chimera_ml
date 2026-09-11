# Training

Run training from an experiment YAML:

~~~bash
chimera-ml validate-config --config-path config.yaml
chimera-ml train --config-path config.yaml
~~~

The CLI loads YAML and seeds the process, requires
experiment_info.params.experiment_name, generates a run name, builds the
datamodule and model, registers them in BuildContext, then builds train config,
loggers, loss, metrics, optimizer, scheduler, and callbacks before calling
Trainer.fit.

Trainer.fit runs epochs epochs. Every training batch performs forward, loss,
finite-value checks, backward, optional gradient clipping, optimizer step, and
optional batch-level scheduler step. Validation loaders run after the train
epoch. Metrics reset at epoch start and their computed values are included in
logs.

## Precision and devices

mixed_precision: true enables torch.amp.autocast and a gradient scaler only
when the resolved device is CUDA. On CPU the loop remains full precision. When
CUDA is unavailable, the requested CUDA device resolves to CPU.

## Loader scheduling

Configure multiple train loaders with:

~~~yaml
train:
  params:
    train_loader_mode: round_robin
    train_stop_on: min
~~~

For weighted, provide train_loader_weights keyed by normalized loader name.
See [Data](data.md) for stop behavior.

## Scheduler

Set use_scheduler: true and provide a scheduler section. With
scheduler_step_per_epoch: false, the scheduler steps after each optimizer
step. A ReduceLROnPlateau-style scheduler receives callback logs when stepped
per epoch and scheduler_monitor selects its value when configured.

## Artifacts

checkpoint_callback writes top-k checkpoints and optionally last.pt.
snapshot_callback can save config and a code.zip source snapshot.
collect_predictions_callback and plotting callbacks use evaluation cache; set
collect_cache: true when using them.
