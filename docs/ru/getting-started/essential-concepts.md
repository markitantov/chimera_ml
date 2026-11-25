# Основные понятия

## Объекты запуска обучения

- **Dataset** возвращает один sample. Плагин решает, как читать пути, признаки, метки и метаданные.
- **DataModule** строит датасеты и предоставляет `train_dataloader()`, `val_dataloader()` и `test_dataloader()`; split может быть одним loader, mapping или sequence.
- **DataLoader** собирает samples и вызывает collate.
- **Collate** превращает список mappings в `Batch`; встроены `masking_collate` и `multimodal_sequence_collate`.
- **Batch** содержит `inputs`, optional `targets`, canonical `masks` и `meta`.
- **Model** получает `Batch` и возвращает `ModelOutput`.
- **ModelOutput** содержит `preds` и необязательный тензор `aux`, например признаки.
- **Loss** превращает output и labeled batch в скалярную функцию потерь.
- **Metrics** накапливают предсказания и целевые значения за эпоху и возвращают именованные скаляры.
- **Optimizer** обновляет параметры модели; **Scheduler** изменяет состояние оптимизатора.
- **Trainer** координирует устройство, прямой и обратный проходы, метрики, журналирование, колбэки и контрольные точки.
- **Callbacks** реагируют на жизненный цикл, сохраняют артефакты или останавливают обучение.
- **Loggers** записывают локальные журналы, метрики и артефакты MLflow.

Фреймворк предоставляет контейнеры, встроенные компоненты, строители, Trainer и CLI. Проект задачи обычно предоставляет Dataset/DataModule и регистрирует модель, функцию потерь и метрики через плагин.

## YAML и реестры

```yaml
model:
  name: tfs_fusion_model
  params:
    input_dims: {audio: 128, video: 256}
    shared_dim: 64
    output_dim: 2
```

`name` должен существовать в нужном registry. Проверяйте keys текущего процесса:

```bash
chimera-ml registry list --type models
chimera-ml registry list --type datamodules
```

Плагины обнаруживаются через группу точек входа `chimera_ml.plugins`. `plugins list` показывает обнаружение, а `registry list` — результат регистрации.
