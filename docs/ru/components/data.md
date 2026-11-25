# Компоненты данных

## Интерфейс Dataset

`Dataset` отвечает за представление отдельных образцов задачи: он читает исходные данные, возвращает входы, целевые значения и метаданные. `DataModule` затем организует эти наборы данных и предоставляет загрузчики для этапов обучения, проверки и тестирования.

Ядро не регистрирует конкретные датасеты: встроенный реестр `DATAMODULES` пуст. Датасет плагина обычно является `torch.utils.data.Dataset`, возвращающим отображение с `inputs`, необязательными `targets`/`target`, `meta` и `masks`:

```python
{
    "inputs": {"audio": torch.randn(10, 80), "video": None},
    "targets": torch.tensor([0, 1]),
    "meta": {"sample_id": "s-1"},
}
```

Набор данных (`Dataset`) отвечает за чтение файлов, извлечение признаков, получение целевых значений и идентификаторов образцов. Функция формирования батчей (`Collate`) объединяет образцы, дополняет последовательности до необходимой длины и формирует канонические маски.

## `DataModule`

`DataModule` принимает `train_dataset`, `val_dataset`, `test_dataset` как один Dataset, отображение именованных датасетов, последовательность датасетов или `None`. Значения по умолчанию: `batch_size=32`, `num_workers=0`, `pin_memory=True`, `persistent_workers=False`, `shuffle_train=True`, `drop_last_train=False`, `collate_fn=MaskingCollate()`.

```python
from chimera_ml.data import DataModule, multimodal_sequence_collate

module = DataModule(
    train_dataset=train_dataset,
    val_dataset={"clean": val_clean, "noisy": val_noisy},
    test_dataset=test_dataset,
    batch_size=16,
    num_workers=2,
    collate_fn=multimodal_sequence_collate(pad_sequences=True),
)
train_loader = module.train_dataloader()
val_loaders = module.val_dataloader()  # mapping сохраняется
```

`train_dataloader`, `val_dataloader`, `test_dataloader` сохраняют форму контейнера: DataLoader, отображение, последовательность или `None`. Параметры только для обучения — shuffle и drop-last. Фабрика DataModule плагина должна принимать `collate_fn`, если используется верхнеуровневый `collate`; одновременная настройка `collate` и `data.params.collate_fn` запрещена.

## Зарегистрированные collate-функции

| Ключ | Когда выбирать | Дополнение и маски |
| --- | --- | --- |
| `masking_collate` | Обычные или устаревшие образцы и общая последовательность | Общая максимальная длина, `sequence_mask` и `<modality>_mask`. |
| `multimodal_sequence_collate` | Независимые временные шкалы модальностей | Отдельное дополнение, `<modality>_mask` и `<modality>_temporal_mask`. |

### `masking_collate`

Параметры: `pad_sequences=True`, `include_legacy_meta_masks=False`, `pad_value=0.0`. Непустые образцы должны иметь совместимые число измерений и размеры признаков. При дополнении `[T,D]` → `[B,T_max,D]`; при `False` формы должны совпадать. Отсутствующие строки заполняются `pad_value`, маска наличия имеет тип float32 и форму `[B]`, `sequence_mask` — тип bool и форму `[B,T_max]`. Целевые значения принимаются из `target` или `targets`; если все отсутствуют, `Batch.targets=None`. Пустой пакет и несовместимые тензоры приводят к `ValueError`.

```python
collate = masking_collate(pad_sequences=True, pad_value=-1.0)
batch = collate([
    {"inputs": {"audio": torch.ones(3, 4)}, "targets": torch.tensor([1.0])},
    {"inputs": {"audio": torch.ones(1, 4)}, "targets": torch.tensor([2.0])},
])
# audio [2, 3, 4], audio_mask [2], sequence_mask [2, 3]
```

### `multimodal_sequence_collate`

Параметры те же. Тензор размерности 1 `[D]` преобразуется в `[B,D]` и временную маску `[B,1]`; тензор размерности 2 `[T,D]` — в независимую форму `[B,T_max,D]`. Отсутствующая строка заполняется нулями; `<modality>_mask` имеет тип float32 и форму `[B]`, `<modality>_temporal_mask` — тип bool и форму `[B,T_max]`. Явные временные маски принимаются из `meta["<modality>_temporal_mask"]`, `meta["masks"]` или масок уровня образца `masks`; длина должна совпадать. Размерности, отличные от 1 и 2, несовместимые ширины, недействительные маски, пустые пакеты и переменная длина при `pad_sequences=False` приводят к `ValueError`.

```python
collate = multimodal_sequence_collate()
batch = collate([
    {"inputs": {"audio": torch.ones(2, 4), "video": torch.ones(3, 6)}},
    {"inputs": {"audio": torch.ones(1, 4)}},
])
# audio [2,2,4], audio_mask [1,1], audio_temporal_mask [2,2]
# video [2,3,6], video_mask [1,0], video_temporal_mask [2,3]
```

Collate-функции не выравнивают временные шкалы аудио и видео. Dataset должен заранее привести кадровые потоки к общей шкале для архитектур вроде RAAV.
