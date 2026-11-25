# Подготовка datasets и loaders

Контракты компонентов и зарегистрированные collates собраны в [каталоге Data](../components/data.md); здесь сохранено workflow-описание подготовки данных.

## Ответственность Dataset

Task Dataset — PyTorch `Dataset` (или plugin equivalent), чей `__getitem__` возвращает mapping. Framework не определяет формат audio, video, text, cached features, labels и IDs.

Практический sample:

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

Разрешён alias `target`. Modality может отсутствовать в sample. `targets=None` допустим для unlabeled data, но `Trainer.fit` отвергает train batch без targets.

## DataModule и splits

Core `DataModule` принимает один dataset, mapping named datasets или sequence для каждого split и строит DataLoaders с `batch_size`, `num_workers`, `pin_memory`, `persistent_workers`, `shuffle_train` и `drop_last_train`:

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

Plugin DataModule обычно создаёт datasets и возвращает тот же interface. Core не регистрирует generic datamodule key, поэтому `data.name` в YAML должен прийти из plugin.

Mapping сохраняет names; sequence получает names вроде `train0`. Multiple train loaders планируются через `train_loader_mode`.

## Batch и canonical masks

`Batch.inputs` — mapping modality к tensors, `targets` — target или `None`, `masks` — canonical flat mapping. `Batch.get_masks()` также понимает legacy `masks["mask"]` и `meta["masks"]`.

- `<modality>_mask` — sample presence; built-in multimodal collate создаёт float32 `[B]`.
- `<modality>_temporal_mask` — valid temporal positions, bool `[B, T]`.
- `sequence_mask` от `masking_collate` — valid padded positions, bool `[B, T_max]`.
- Missing inputs заполняются нулями, но masks отличают missing data от настоящего нуля.

## Выбор collate

### `masking_collate`

Используйте для legacy/general behavior. Он дополняет первую размерность при `pad_sequences: true`, создаёт `sequence_mask` и `<modality>_mask`.

```yaml
collate:
  name: masking_collate
  params:
    pad_sequences: true
    pad_value: 0.0
    include_legacy_meta_masks: false
```

Sequences `audio` длины 3 и 1 дают `audio.shape == [2, 3, D]`, `audio_mask.shape == [2]`, `sequence_mask.shape == [2, 3]`.

### `multimodal_sequence_collate`

Используйте для independent feature timelines. `[D]` становится `[B, D]` с temporal mask `[B, 1]`; `[T, D]` отдельно для каждой modality дополняется до `[B, T_max, D]`.

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

Explicit `<modality>_temporal_mask` может быть в `meta`, `meta["masks"]` или sample-level `masks`; длина должна совпадать с sequence. Поддерживаются `pad_value`, `pad_sequences`, `include_legacy_meta_masks`. Timelines автоматически не выравниваются.

Top-level `collate` используется CLI `train` и `eval`; DataModule factory должна явно принимать keyword `collate_fn`. Одновременные `collate:` и `data.params.collate_fn` запрещены.
