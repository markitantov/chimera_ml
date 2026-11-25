# Обзор компонентов

CHIMERA-ML разделяет публичные контракты и зарегистрированные factory. `Batch`, `ModelOutput`, `BaseModel` и `BaseMetric` задают интерфейсы вызовов, а registry key делает factory доступной из YAML. Реализации datasets и task-specific prediction steps обычно находятся в plugins.

| Категория | Назначение | Публичная абстракция | Построение |
| --- | --- | --- | --- |
| Core | Batch, output, config и metadata построения | `Batch`, `ModelOutput`, `ExperimentConfig`, `BuildContext` | Python и builders |
| Data | Dataset, loaders и collates | `DataModule`, `MaskingCollate`, `MultimodalSequenceCollate` | `DATAMODULES` / `COLLATES` |
| Models | Batch → ModelOutput | `BaseModel` | `MODELS` или Python |
| Training | Loss, metrics, optimizer, scheduler | `BaseLoss`, `BaseMetric` и factories | `LOSSES`, `METRICS`, `OPTIMIZERS`, `SCHEDULERS` |
| Callbacks | Lifecycle Trainer, state и artifacts | `BaseCallback` | `CALLBACKS` |
| Logging | Console/file и MLflow | `BaseLogger` и logger-compatible objects | `LOGGERS` |
| Inference | Pipeline steps и XAI | `InferenceContext` | `INFERENCE_STEPS` |

Типичный training flow: `Dataset → DataModule → DataLoader → collate → Batch → BaseModel → ModelOutput → loss/metrics → optimizer/scheduler`; callbacks и loggers наблюдают Trainer. Inference использует отдельный `InferenceContext`, а не `BuildContext`.

- [Core](core.md): `Batch`, `ModelOutput`, configuration и builders.
- [Data](data.md): dataset responsibilities, DataModule и collates.
- [Models](models.md): контракт модели и четыре fusion architectures.
- [Training components](training.md): losses, metrics, optimizers и schedulers.
- [Callbacks](callbacks/index.md): lifecycle и все зарегистрированные callbacks.
- [Loggers](logging.md): console/file и MLflow.

Проверяйте живой registry командой `chimera-ml registry list`. Plugin keys появляются только после установки и discovery plugin.
