# Обзор обучения

Обучение CHIMERA-ML — конфигурационный workflow `Trainer.fit`:

```text
Dataset -> DataModule -> DataLoader -> Collate -> Batch -> Model
        -> ModelOutput -> Loss/Metrics -> Optimizer -> Trainer
        -> Callbacks/Loggers -> Checkpoints
                           \-> Scheduler
```

## Порядок работы

1. Plugin Dataset читает sample с modality inputs, targets и metadata.
2. DataModule строит train/validation/test DataLoaders и может вернуть named loaders.
3. Collate превращает samples в `Batch`, дополняет sequences и создаёт masks.
4. Trainer переносит batch на device и вызывает `model(batch)`.
5. Loss получает `ModelOutput` и `Batch`, metrics обновляются по labeled batch.
6. Training делает backward, optional gradient clipping, optimizer step и scheduler step.
7. После train epoch запускается validation; callbacks получают flat logs вроде `train/loss`, `val/loss` и metric keys.
8. Loggers сохраняют logs/metrics/artifacts, checkpoint callback пишет model state.

Core предоставляет `Batch`, `ModelOutput`, `DataModule`, collates, Trainer, builders, built-in components и CLI. Project supplies task data и обычно регистрирует DataModule, model, loss и metrics. Plugin factory signatures определяют plugin-specific params.

## Команды

```bash
chimera-ml validate-config --config-path config.yaml
chimera-ml train --config-path config.yaml
chimera-ml eval --config-path config.yaml --checkpoint-path logs/.../checkpoints/last.pt
```

Training обновляет параметры; evaluation не делает optimizer updates. Deployment inference — отдельный pipeline во вкладке Inference.
