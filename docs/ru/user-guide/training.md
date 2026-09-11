# Training

Запустите обучение из experiment YAML:

~~~bash
chimera-ml validate-config --config-path config.yaml
chimera-ml train --config-path config.yaml
~~~

CLI загружает YAML и seed, требует experiment_info.params.experiment_name,
создаёт run name, строит datamodule/model и регистрирует их в BuildContext,
затем строит train config, loggers, loss, metrics, optimizer, scheduler и
callbacks и вызывает Trainer.fit.

Trainer.fit выполняет epochs. Каждый train batch проходит forward, loss,
finite-value checks, backward, optional gradient clipping, optimizer step и
optional scheduler step. Validation loaders запускаются после train epoch.

## Precision и devices

mixed_precision: true включает AMP и scaler только на CUDA. На CPU цикл
остаётся full precision. Если CUDA недоступна, requested CUDA device заменяется
на CPU.

## Loader scheduling

~~~yaml
train:
  params:
    train_loader_mode: round_robin
    train_stop_on: min
~~~

Для weighted задайте train_loader_weights по имени loader. Подробности в
[Data](data.md).

## Scheduler

Задайте use_scheduler: true и секцию scheduler. При
scheduler_step_per_epoch: false scheduler шагает после каждого optimizer step.
ReduceLROnPlateau получает callback logs при epoch stepping; scheduler_monitor
задаёт monitor, если он настроен.

## Artifacts

checkpoint_callback пишет top-k и optional last.pt; snapshot_callback может
сохранить config и code.zip. Callbacks predictions/plots используют evaluation
cache, поэтому оставляйте collect_cache: true.
