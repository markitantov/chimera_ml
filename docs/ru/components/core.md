# Компоненты ядра

## Публичные импорты

Ядро экспортирует `Batch`, `ModelOutput`, `ExperimentConfig`, `load_yaml`, `Registry` и именованные реестры. Подсистема данных экспортирует `DataModule`, оба класса и фабрики collate-функций, а также утилиты загрузчиков; подсистема моделей экспортирует `BaseModel` и четыре класса и фабрики архитектур.

```python
from chimera_ml.core import Batch, ModelOutput, MODELS, Registry
from chimera_ml.data import DataModule, MaskingCollate, MultimodalSequenceCollate
from chimera_ml.models import BaseModel, TFSFusionModel
```

## `Batch`

`Batch` — публичный контейнер, который collate передаёт модели:

```python
Batch(
    inputs={"audio": audio_tensor, "video": video_tensor},
    targets=target_tensor_or_none,
    masks={"audio_mask": presence, "audio_temporal_mask": valid_positions},
    meta={"sample_meta": [...]},
)
```

- `inputs` связывает стабильные имена модальностей с тензорами; размерность пакета находится на оси 0, а модальность может отсутствовать.
- `targets` — целевое значение задачи или `None` для теста и инференса без меток.
- `masks` — каноническое плоское отображение: `<modality>_mask` для наличия, `<modality>_temporal_mask` для `[B,T]`, `sequence_mask` для общей маски дополнения.
- `meta` хранит идентификаторы, длины и прикладные метаданные, но не заменяет канонические маски.

`get_masks()` возвращает нормализованное отображение, а `get_masks("audio_mask")` — одну маску. Для обратной совместимости читаются вложенные `Batch.masks["mask"]` и `meta["masks"]`; `{ "mask": {"audio": value} }` превращается в `audio_mask`. Модель сама определяет, какие модальности и маски поддерживает.

```python
# Unimodal labeled vector
Batch(inputs={"text": torch.randn(8, 768)}, targets=torch.randint(0, 3, (8,)))

# Multimodal temporal features и отсутствующий audio у одного sample
Batch(
    inputs={"audio": torch.randn(4, 12, 80), "video": torch.randn(4, 12, 512)},
    targets=torch.randn(4, 2),
    masks={"audio_mask": torch.tensor([1, 1, 0, 1]),
           "audio_temporal_mask": torch.ones(4, 12, dtype=torch.bool)},
)

# Unlabeled inference
Batch(inputs={"audio": torch.randn(1, 20, 80)}, targets=None)
```

Заполненный нулями отсутствующий вход не равен настоящему нулевому сигналу; различие задают маски наличия и временные маски.

## `ModelOutput`

`ModelOutput` передаёт результат работы модели дальше в Trainer, функции потерь, метрики и шаги инференса. Поле `preds` содержит основное предсказание, а `aux` предназначено для дополнительных представлений и диагностики.

`ModelOutput(preds: torch.Tensor, aux: dict[str, torch.Tensor] | None = None)` — публичный результат модели. `preds` содержит логиты для классификации, логиты для multi-label BCE или значения регрессии; точная форма определяется задачей и моделью. `aux` содержит необязательные тензоры: объединённые признаки, веса внимания, маски валидности и диагностические данные.

Trainer вычисляет функцию потерь по `output.preds` и `batch.targets`, а значения `aux` автоматически не участвуют в оптимизации. XAI копирует только явно запрошенные ключи `aux`, выровненные по образцам пакета. Имена `aux` зависят от архитектуры.

## Конфигурация и строители

YAML обучения использует отображение `{"name": "mse_loss", "params": {"reduction": "mean"}}`. `build_from_registry` приводит имя к нижнему регистру, проверяет отображение `params`, объединяет внедрённые зависимости времени выполнения и вызывает `factory(**kwargs)`. `build_optimizer` внедряет `model`, планировщик получает `optimizer`, а необязательный `context` передаётся только фабрике, которая его принимает.

Публичные строители: `build_model`, `build_loss`, `build_metrics`, `build_datamodule`, `build_collate`, `build_optimizer`, `build_scheduler`, `build_callbacks`, `build_logger`. Параметры фабрики не обязательно совпадают с параметрами базового класса PyTorch.
