# Компоненты данных

## Интерфейс Dataset

Core не регистрирует concrete dataset: в built-in registry `DATAMODULES` пуст. Plugin dataset обычно является `torch.utils.data.Dataset`, возвращающим mapping с `inputs`, optional `targets`/`target`, `meta` и `masks`:

```python
{
    "inputs": {"audio": torch.randn(10, 80), "video": None},
    "targets": torch.tensor([0, 1]),
    "meta": {"sample_id": "s-1"},
}
```

Dataset отвечает за file format, feature extraction, labels и IDs; collate — за batching, padding и canonical masks.

## `DataModule`

`DataModule` принимает `train_dataset`, `val_dataset`, `test_dataset` как один Dataset, mapping named datasets, sequence datasets или `None`. Defaults: `batch_size=32`, `num_workers=0`, `pin_memory=True`, `persistent_workers=False`, `shuffle_train=True`, `drop_last_train=False`, `collate_fn=MaskingCollate()`.

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

`train_dataloader`, `val_dataloader`, `test_dataloader` сохраняют форму container: DataLoader, mapping, sequence или `None`. Train-only options — shuffle/drop-last. Plugin DataModule factory должна принимать `collate_fn`, если используется top-level `collate`; одновременная настройка `collate` и `data.params.collate_fn` запрещена.

## Зарегистрированные collate-функции

| Key | Когда выбирать | Padding и masks |
| --- | --- | --- |
| `masking_collate` | General/legacy samples и shared sequence | Общая максимальная длина, `sequence_mask` и `<modality>_mask`. |
| `multimodal_sequence_collate` | Независимые timelines модальностей | Отдельный padding, `<modality>_mask` и `<modality>_temporal_mask`. |

### `masking_collate`

Parameters: `pad_sequences=True`, `include_legacy_meta_masks=False`, `pad_value=0.0`. Non-empty samples должны иметь совместимые rank и feature dimensions. При padding `[T,D]` → `[B,T_max,D]`; при `False` формы должны совпадать. Missing rows заполняются `pad_value`, presence — float32 `[B]`, `sequence_mask` — bool `[B,T_max]`. Targets принимают `target` или `targets`; если все отсутствуют, `Batch.targets=None`. Empty batch и несовместимые tensors дают `ValueError`.

```python
collate = masking_collate(pad_sequences=True, pad_value=-1.0)
batch = collate([
    {"inputs": {"audio": torch.ones(3, 4)}, "targets": torch.tensor([1.0])},
    {"inputs": {"audio": torch.ones(1, 4)}, "targets": torch.tensor([2.0])},
])
# audio [2, 3, 4], audio_mask [2], sequence_mask [2, 3]
```

### `multimodal_sequence_collate`

Parameters те же. Rank-one `[D]` → `[B,D]` и temporal mask `[B,1]`; rank-two `[T,D]` → independent `[B,T_max,D]`. Missing row zero-filled; `<modality>_mask` — float32 `[B]`, `<modality>_temporal_mask` — bool `[B,T_max]`. Explicit temporal masks принимаются из sample `meta["<modality>_temporal_mask"]`, `meta["masks"]` или sample-level `masks`; длина должна совпадать. Ranks кроме 1/2, incompatible widths, invalid masks, empty batches и variable lengths with `pad_sequences=False` дают `ValueError`.

```python
collate = multimodal_sequence_collate()
batch = collate([
    {"inputs": {"audio": torch.ones(2, 4), "video": torch.ones(3, 6)}},
    {"inputs": {"audio": torch.ones(1, 4)}},
])
# audio [2,2,4], audio_mask [1,1], audio_temporal_mask [2,2]
# video [2,3,6], video_mask [1,0], video_temporal_mask [2,3]
```

Collates не выравнивают timelines audio/video. Dataset должен заранее привести frame streams к общей шкале для архитектур вроде RAAV.
