# Основные понятия

## Объекты training run

- **Dataset** возвращает один sample. Plugin решает, как читать paths, features, labels и metadata.
- **DataModule** строит datasets и предоставляет `train_dataloader()`, `val_dataloader()` и `test_dataloader()`; split может быть одним loader, mapping или sequence.
- **DataLoader** собирает samples и вызывает collate.
- **Collate** превращает список mappings в `Batch`; встроены `masking_collate` и `multimodal_sequence_collate`.
- **Batch** содержит `inputs`, optional `targets`, canonical `masks` и `meta`.
- **Model** получает `Batch` и возвращает `ModelOutput`.
- **ModelOutput** содержит `preds` и optional tensor `aux`, например features.
- **Loss** превращает output и labeled batch в scalar loss.
- **Metrics** накапливают predictions/targets за epoch и возвращают named scalars.
- **Optimizer** обновляет параметры модели; **Scheduler** меняет optimizer state.
- **Trainer** координирует device, forward/loss/backward, metrics, logging, callbacks и checkpoints.
- **Callbacks** реагируют на lifecycle и сохраняют artifacts или останавливают training.
- **Loggers** пишут local logs и MLflow metrics/artifacts.

Framework предоставляет containers, built-in components, builders, Trainer и CLI. Task project обычно предоставляет Dataset/DataModule и регистрирует model, loss и metrics через plugin.

## YAML и registries

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

Plugins обнаруживаются через entry-point group `chimera_ml.plugins`. `plugins list` показывает discovery, `registry list` — результат registration.
