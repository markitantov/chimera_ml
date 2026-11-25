# Обзор компонентов

CHIMERA-ML разделяет публичные контракты и зарегистрированные фабрики. `Batch`, `ModelOutput`, `BaseModel` и `BaseMetric` задают интерфейсы вызовов, а ключ реестра делает фабрику доступной из YAML. Реализации датасетов и шагов предсказания, специфичных для задачи, обычно находятся в плагинах.

| Категория | Назначение | Публичная абстракция | Построение |
| --- | --- | --- | --- |
| Ядро | Batch, результат, конфигурация и метаданные построения | `Batch`, `ModelOutput`, `ExperimentConfig`, `BuildContext` | Python и строители |
| Данные | Dataset, загрузчики и collate-функции | `DataModule`, `MaskingCollate`, `MultimodalSequenceCollate` | `DATAMODULES` / `COLLATES` |
| Модели | Преобразование Batch в ModelOutput | `BaseModel` | `MODELS` или Python |
| Обучение | Функции потерь, метрики, оптимизаторы и планировщики | `BaseLoss`, `BaseMetric` и фабрики | `LOSSES`, `METRICS`, `OPTIMIZERS`, `SCHEDULERS` |
| Колбэки | Жизненный цикл Trainer, состояние и артефакты | `BaseCallback` | `CALLBACKS` |
| Журналирование | Вывод в консоль, файл и MLflow | `BaseLogger` и совместимые с журналированием объекты | `LOGGERS` |
| Инференс | Шаги конвейера и XAI | `InferenceContext` | `INFERENCE_STEPS` |

Типичный поток обучения выглядит так: `Dataset → DataModule → DataLoader → collate → Batch → BaseModel → ModelOutput → loss/metrics → optimizer/scheduler`. Колбэки и журналировщики наблюдают за Trainer. Инференс использует отдельный `InferenceContext`, а не `BuildContext`.

- [Ядро](core.md): `Batch`, `ModelOutput`, конфигурация и строители.
- [Данные](data.md): ответственность датасета, DataModule и collate-функции.
- [Модели](models.md): контракт модели и четыре архитектуры слияния.
- [Компоненты обучения](training.md): функции потерь, метрики, оптимизаторы и планировщики.
- [Колбэки](callbacks/index.md): жизненный цикл и все зарегистрированные колбэки.
- [Журналирование](logging.md): вывод в консоль, файл и MLflow.

Проверяйте содержимое реестра командой `chimera-ml registry list`. Ключи плагина появляются только после его установки и обнаружения.
