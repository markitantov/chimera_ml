# Подготовка датасетов и загрузчиков

Контракты компонентов и зарегистрированные collate-функции собраны в [каталоге данных](../components/data.md); здесь описан сценарий подготовки данных.

## Ответственность Dataset

Набор данных задачи — PyTorch `Dataset` (или эквивалент плагина), чей `__getitem__` возвращает отображение. Фреймворк не определяет формат аудио, видео, текста, кэшированных признаков, меток и идентификаторов.

Пример образца:

```python
{
    "inputs": {
        "audio": torch.Tensor,  # [T_audio, D_audio] или [D_audio]
        "video": torch.Tensor,  # [T_video, D_video] или [D_video]
    },
    "targets": torch.Tensor,     # [2] или [T_target, 2]
    "meta": {"id": "sample-001"},
}
```

Разрешён псевдоним `target`. Модальность может отсутствовать в образце. `targets=None` допустим для данных без меток, но `Trainer.fit` отклоняет обучающий батч без целевых значений.

## DataModule и выборки

Ядро `DataModule` принимает один набор данных, отображение именованных наборов или последовательность для каждой выборки и строит DataLoader с `batch_size`, `num_workers`, `pin_memory`, `persistent_workers`, `shuffle_train` и `drop_last_train`:

```python
from chimera_ml.data import DataModule, multimodal_sequence_collate

module = DataModule(
    train_dataset=train_dataset,
    val_dataset={"validation": validation_dataset},
    test_dataset=test_dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    collate_fn=multimodal_sequence_collate(),
)
```

DataModule плагина обычно создаёт наборы данных и возвращает тот же интерфейс. Ядро не регистрирует универсальный ключ DataModule, поэтому `data.name` в YAML должен быть предоставлен плагином.

Отображение сохраняет имена; последовательность получает имена вроде `train0`. Несколько обучающих загрузчиков планируются через `train_loader_mode`.

## Batch и канонические маски

`Batch.inputs` — отображение модальности на тензоры, `targets` — целевое значение или `None`, `masks` — каноническое плоское отображение. `Batch.get_masks()` также понимает вложенные маски обратной совместимости `masks["mask"]` и `meta["masks"]`.

- `<modality>_mask` — наличие модальности в образце; встроенная мультимодальная collate-функция создаёт float32 `[B]`.
- `<modality>_temporal_mask` — допустимые временные позиции, bool `[B, T]`.
- `sequence_mask` от `masking_collate` — допустимые позиции после дополнения, bool `[B, T_max]`.
- Отсутствующие входы заполняются нулями, но маски отличают отсутствующие данные от настоящего нуля.

## Выбор collate-функции

### `masking_collate`

Используйте для совместимости с устаревшим и общим поведением. Он дополняет первую размерность при `pad_sequences: true`, создаёт `sequence_mask` и `<modality>_mask`.

```yaml
collate:
  name: masking_collate
  params:
    pad_sequences: true
    pad_value: 0.0
    include_legacy_meta_masks: false
```

Последовательности `audio` длины 3 и 1 дают `audio.shape == [2, 3, D]`, `audio_mask.shape == [2]`, `sequence_mask.shape == [2, 3]`.

### `multimodal_sequence_collate`

Используйте для независимых временных шкал признаков. `[D]` становится `[B, D]` с временной маской `[B, 1]`; `[T, D]` отдельно для каждой модальности дополняется до `[B, T_max, D]`.

```python
import torch
from chimera_ml.data.multimodal_sequence_collate import MultimodalSequenceCollate

collate = MultimodalSequenceCollate(pad_sequences=True, pad_value=0.0)
batch = collate([
    {"inputs": {"audio": torch.ones(2, 4), "video": torch.ones(3, 6)}, "targets": torch.tensor([1, 0])},
    {"inputs": {"audio": torch.ones(1, 4)}, "targets": torch.tensor([0, 1]), "meta": {"id": "s2"}},
])
assert batch.inputs["audio"].shape == (2, 2, 4)
assert batch.inputs["video"].shape == (2, 3, 6)
assert batch.get_masks("audio_temporal_mask").shape == (2, 2)
assert batch.get_masks("video_temporal_mask").shape == (2, 3)
```

Явная `<modality>_temporal_mask` может находиться в `meta`, `meta["masks"]` или масках уровня образца `masks`; длина должна совпадать с последовательностью. Поддерживаются `pad_value`, `pad_sequences`, `include_legacy_meta_masks`. Временные шкалы автоматически не выравниваются.

Верхнеуровневый `collate` используется CLI `train` и `eval`; фабрика DataModule должна явно принимать именованный параметр `collate_fn`. Одновременная настройка `collate:` и `data.params.collate_fn` запрещена.
