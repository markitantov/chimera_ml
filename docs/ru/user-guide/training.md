# Запуск обучения

## Сначала проверка

```bash
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml train --config-path examples/va_estimation/configs/multimodal_train.yaml
```

`validate-config` загружает YAML и проверяет разделы фреймворка без создания датасетов и запуска обучения. `train` загружает плагины, задаёт начальное состояние, требует `experiment_info.params.experiment_name`, строит компоненты и вызывает `Trainer.fit`.

## Что делает `Trainer.fit`

Перед началом цикла обучения Trainer один раз вызывает у колбэков `on_fit_start`. Затем на каждой эпохе он:

1. вызывает `on_epoch_start`;
2. перебирает обучающие батчи, переносит их на устройство, вызывает модель, вычисляет функцию потерь, проверяет конечность значений, выполняет обратное распространение, при необходимости ограничивает градиенты, делает шаг оптимизатора, вызывает применимые события обработки батча и обновляет метрики качества;
3. при пошаговом обновлении вызывает планировщик после обновления оптимизатора;
4. запускает загрузчики валидационной выборки без вычисления градиентов и обновляет метрики и кэш;
5. записывает метрики обучения и проверки и скорости обучения, вызывает `on_epoch_end`, а затем при пошаговом обновлении по эпохам вызывает планировщик.

Если колбэк установил `trainer.stop_training`, Trainer прекращает цикл и после нормального завершения или досрочной остановки обучения вызывает `on_fit_end`.

Для обучения требуется `batch.targets`. Для данных без меток используйте оценку или инференс.

## Устройство, точность и ограничение градиентов

`device: cuda` переходит на CPU при недоступной CUDA. `mixed_precision: true` включает AMP и масштабирование градиентов только на CUDA. `grad_clip_norm` задаёт `clip_grad_norm_`, а `null` отключает ограничение градиентов.

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

`single` — режим по умолчанию. `round_robin` чередует загрузчики, `weighted` использует `train_loader_weights`. `min` завершает смешанную эпоху при первом исчерпавшемся загрузчике, `max` — при последнем. Веса должны быть положительными и соответствовать нормализованным именам загрузчиков.

## Оптимизатор и планировщик

Встроенные ключи оптимизаторов: `adamw_optimizer`, `adam_optimizer`, `sgd_optimizer`. Ключи планировщиков: `steplr_scheduler`, `cosineannealinglr_scheduler`, `reduceonplateau_scheduler`; параметры передаются соответствующему планировщику PyTorch.

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

При `scheduler_step_per_epoch: false` планировщик выполняет шаг после каждого обновления оптимизатора. При обновлении по эпохам планировщик, управляемый метрикой, получает словарь метрик с ключами, предварёнными именем выборки, и `scheduler_monitor`. Одной секции `scheduler` недостаточно: нужен `use_scheduler: true`.

## Метрики, колбэки и результаты

Trainer добавляет префикс выборки к ключам метрик: метрика `mae` становится `train/mae` или `val/mae`; колбэк плагина может создать `val_framewise/va_ccc_metric`. Именно этот плоский ключ указывайте в `monitor`.

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

Конфигурацию и `code.zip` создаёт колбэк снимка при соответствующих флагах. Контрольная точка содержит эпоху, глобальный шаг, состояния модели и оптимизатора, а также состояние планировщика, если он есть. Колбэк контрольных точек сохраняет `last.pt` и лучшие результаты под наблюдением согласно `mode` и `save_top_k`.

## Частые ошибки

- Нет `experiment_info.params.experiment_name`: добавьте его для `train`/`sweep`.
- Неизвестный ключ реестра: установите плагин и проверьте `plugins list`/`registry list`.
- Нет целевых значений: исправьте Dataset или collate-функцию.
- Монитор не найден: используйте точный ключ с префиксом из журналов.
- Ошибка формы или нечислового значения: сравните формы модальностей и целевых значений с контрактом модели и функции потерь.
