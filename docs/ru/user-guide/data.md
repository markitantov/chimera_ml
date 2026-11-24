# Data и loaders

Datamodule factory возвращает train, validation и test DataLoaders. Dataset
передаёт тензоры признаков и может добавлять targets, masks и metadata.

## Collates и маски

В CHIMERA-ML есть два разных встроенных collate.

### MaskingCollate

MaskingCollate с registry key masking_collate сохраняет legacy-поведение для
существующих pipelines. Он дополняет или объединяет tensor fields, создаёт
sequence_mask и маски присутствия вроде audio_mask, а при настройке может
копировать маски в legacy meta["masks"]. Новый collate его не заменяет.

### MultimodalSequenceCollate

MultimodalSequenceCollate с registry key multimodal_sequence_collate нужен для
модальностей признаков с независимыми длинами последовательностей.

- Вектор образца [D] собирается в [B, D], временная маска имеет форму [B, 1].
- Последовательность [T, D] отдельно для каждой модальности дополняется до
  [B, T_max, D].
- Для каждой модальности создаётся float32-маска присутствия
  `<modality>_mask` формы [B].
- Для временной модальности создаётся булева маска
  `<modality>_temporal_mask` формы [B, T_max].
- Отсутствующие строки заполняются нулями, для дополнения используется
  pad_value.
- Явная одномерная временная маска может находиться в
  `meta[<modality>_temporal_mask]`, `meta["masks"][<modality>_temporal_mask]`
  или в `sample["masks"][<modality>_temporal_mask]`; она проверяется
  и объединяется с маской дополнения.
- Настройки: pad_sequences, pad_value и include_legacy_meta_masks.
- Поддерживаются последовательности нулевой длины.
- Автоматического выравнивания временных шкал между модальностями нет.

Например, последовательности audio длины 2 и 1 получают audio [B, 2, D_audio],
а video длины 3 и 2 — video [B, 3, D_video]. Каждая модальность сохраняет
собственную ширину последовательности и маску; общий максимум для них не
используется.

Минимальная конфигурация через registry:

~~~yaml
collate:
  name: multimodal_sequence_collate
  params:
    pad_sequences: true
~~~

Во время CLI `train` и `eval` верхнеуровневая секция передаётся через
`build_collate()` в factory выбранного datamodule как `collate_fn`. Custom
datamodule должен явно объявить keyword-параметр `collate_fn`; неподдерживаемый
ход завершится понятной ошибкой, а не тихим игнорированием настройки. Если
секции нет, CLI не создаёт и не внедряет collate, поэтому сохраняется текущий
default datamodule. Не задавайте одновременно эту секцию и
`data.params.collate_fn`.

Fusion utilities получают эти маски через _mask(). Выделенная временная маска
имеет приоритет и проверяется относительно нормализованных признаков [B, T, D].
Старые форматы [B], [B, T] и совместимый sequence_mask остаются поддержанными.

## Multiple loaders

Loader может быть одним DataLoader, mapping или list/tuple. Training
поддерживает single, round_robin и weighted modes. Имена split нормализуются
для стабильных metric keys.

## Batch contract

~~~python
Batch(
    inputs={"audio": audio_tensor, "video": video_tensor},
    targets=target_tensor,
    masks={"audio_temporal_mask": audio_temporal_mask},
    meta={"id": sample_id},
)
~~~

Batch.get_masks() возвращает все маски, а с аргументом — одну плоскую маску.
Для совместимости читаются также legacy nested masks из masks["mask"] и
meta["masks"].
