# Обзор обучения

Обучение CHIMERA-ML — конфигурационный сценарий `Trainer.fit`:

```text
Dataset -> DataModule -> DataLoader -> Collate -> Batch -> Model
        -> ModelOutput -> Loss/Metrics -> Optimizer -> Trainer
        -> Callbacks/Loggers -> Checkpoints
                           \-> Scheduler
```

## Порядок работы

1. Датасет плагина читает образцы с входами модальностей, целевыми значениями и метаданными.
2. DataModule строит загрузчики train/validation/test и может вернуть именованные загрузчики.
3. Collate превращает образцы в `Batch`, дополняет последовательности и создаёт маски.
4. Trainer переносит пакет на устройство и вызывает `model(batch)`.
5. Функция потерь получает `ModelOutput` и `Batch`, а метрики обновляются по размеченному пакету.
6. Обучение выполняет обратное распространение, при необходимости ограничивает градиенты, а затем делает шаг оптимизатора и планировщика.
7. После train epoch запускается validation; callbacks получают flat logs вроде `train/loss`, `val/loss` и metric keys.
8. Логгеры сохраняют журналы, метрики и артефакты, а колбэк контрольных точек записывает состояние модели.

Core предоставляет `Batch`, `ModelOutput`, `DataModule`, collate-функции, Trainer, builders, встроенные компоненты и CLI. Проект предоставляет данные задачи и обычно регистрирует DataModule, модель, функцию потерь и метрики. Сигнатуры фабрик плагина определяют параметры, специфичные для плагина.

## Команды

```bash
chimera-ml validate-config --config-path config.yaml
chimera-ml train --config-path config.yaml
chimera-ml eval --config-path config.yaml --checkpoint-path logs/.../checkpoints/last.pt
```

Обучение обновляет параметры; оценка не выполняет шагов оптимизатора. Вывод для развёртывания — отдельный конвейер во вкладке «Вывод».
