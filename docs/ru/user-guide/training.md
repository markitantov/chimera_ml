# Запуск обучения

## Сначала проверка

```bash
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml train --config-path examples/va_estimation/configs/multimodal_train.yaml
```

`validate-config` загружает YAML и проверяет разделы фреймворка без создания датасетов и обучения. `train` загружает plugins, задаёт seed, требует `experiment_info.params.experiment_name`, строит компоненты и вызывает `Trainer.fit`.

## Что делает `Trainer.fit`

На каждой эпохе Trainer:

1. вызывает `on_fit_start` и `on_epoch_start` callbacks;
2. перебирает train batches, переносит их на device, вызывает model, считает loss, проверяет finite values, делает backward, при необходимости ограничивает градиенты, optimizer step и обновляет metrics;
3. при batch stepping вызывает scheduler после optimizer update;
4. запускает validation loaders без gradients и обновляет metrics/cache;
5. пишет train/validation metrics и learning rates, вызывает `on_epoch_end`, затем при epoch stepping вызывает scheduler;
6. останавливается, если callback установил `trainer.stop_training`, и вызывает `on_fit_end`.

Для обучения требуется `batch.targets`. Для данных без меток используйте оценку или инференс.

## Устройство, точность и ограничение градиентов

`device: cuda` переходит на CPU при недоступной CUDA. `mixed_precision: true` включает AMP и scaler только на CUDA. `grad_clip_norm` задаёт `clip_grad_norm_`, `null` отключает clipping.

```yaml
train:
  params:
    epochs: 10
    device: cuda
    mixed_precision: true
    grad_clip_norm: 1.0
```

## Несколько загрузчиков

```yaml
train:
  params:
    train_loader_mode: round_robin
    train_stop_on: min
```

`single` — default. `round_robin` чередует loaders, `weighted` использует `train_loader_weights`. `min` заканчивает mixed epoch при первом exhausted loader, `max` — при последнем. Weights должны быть положительными и соответствовать normalized loader names.

## Оптимизатор и планировщик

Встроенные ключи оптимизаторов: `adamw_optimizer`, `adam_optimizer`, `sgd_optimizer`. Scheduler keys: `steplr_scheduler`, `cosineannealinglr_scheduler`, `reduceonplateau_scheduler`; params передаются соответствующему PyTorch scheduler.

```yaml
optimizer:
  name: adamw_optimizer
  params: {lr: 0.001, weight_decay: 0.01}
scheduler:
  name: reduceonplateau_scheduler
  params: {factor: 0.5, patience: 2}
train:
  params:
    use_scheduler: true
    scheduler_step_per_epoch: true
    scheduler_monitor: val/loss
```

При `scheduler_step_per_epoch: false` планировщик выполняет шаг после каждого обновления оптимизатора. При epoch stepping metric-driven scheduler получает flat callback logs и `scheduler_monitor`. Одной секции scheduler недостаточно: нужен `use_scheduler: true`.

## Метрики, колбэки и результаты

Trainer добавляет префикс раздела к ключам метрик: metric `mae` становится `train/mae` или `val/mae`; plugin callback может создать `val_framewise/va_ccc_metric`. Именно этот flat key указывайте в monitor.

При `log_path: logs` и `experiment_name: va_estimation_fusion`:

```text
logs/<experiment>/<run>/
├── train.log
├── checkpoints/
│   ├── last.pt
│   └── epoch=...pt
├── config.yaml
└── code.zip
```

Конфигурацию и `code.zip` создаёт колбэк снимка при соответствующих flags. Checkpoint содержит epoch, global step, model/optimizer state и scheduler state, если он есть. Checkpoint callback сохраняет `last.pt` и monitored improvements согласно `mode`/`save_top_k`.

## Частые ошибки

- Нет `experiment_info.params.experiment_name`: добавьте его для `train`/`sweep`.
- Неизвестный ключ реестра: установите plugin и проверьте `plugins list`/`registry list`.
- Нет целевых значений: исправьте Dataset/collate.
- Monitor not found: используйте точный prefixed key из logs.
- Ошибка формы или нечислового значения: сравните modality/target shapes с model/loss contract.
