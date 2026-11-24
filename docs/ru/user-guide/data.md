# Data и loaders

Datamodule factory возвращает train, validation и test DataLoaders. Dataset
передаёт feature tensors и может добавлять targets, masks и metadata.

## Collates и masks

В CHIMERA-ML есть два разных встроенных collate.

### MaskingCollate

MaskingCollate с registry key masking_collate сохраняет legacy behavior для
существующих pipelines. Он padding-ит или stack-ит tensor fields, создаёт
sequence_mask и presence masks вроде audio_mask, а при настройке может копировать
masks в legacy meta["masks"]. Новый collate его не заменяет.

### MultimodalSequenceCollate

MultimodalSequenceCollate с registry key multimodal_sequence_collate нужен для
feature modalities с независимыми temporal lengths.

- Вектор sample [D] собирается в [B, D], temporal mask имеет форму [B, 1].
- Sequence [T, D] отдельно для каждой modality padding-ится в [B, T_max, D].
- Для каждой modality создаётся float32 presence mask `<modality>_mask` формы [B].
- Для temporal modality создаётся boolean `<modality>_temporal_mask` формы
  [B, T_max] и имеет boolean dtype. Presence mask имеет float32 dtype.
- Отсутствующие rows заполняются нулями, padding использует pad_value.
- Явная one-dimensional temporal mask может быть в
  meta[`<modality>_temporal_mask`], meta[masks][`<modality>_temporal_mask`] или
  sample-level masks[`<modality>_temporal_mask`]; она проверяется и объединяется
  с padding mask.
- Настройки: pad_sequences, pad_value и include_legacy_meta_masks.
- Поддерживаются zero-length sequences.
- Автоматического выравнивания timelines между modalities нет.

Например, audio sequences длины 2 и 1 получают audio [B, 2, D_audio], а video
длины 3 и 2 — video [B, 3, D_video]. Каждая modality сохраняет собственную
temporal width и mask; общий maximum между ними не используется.

Минимальная registry configuration:

~~~yaml
collate:
  name: multimodal_sequence_collate
  params:
    pad_sequences: true
~~~

Fusion utilities получают эти masks через _mask(). Dedicated temporal mask имеет
приоритет и проверяется относительно normalized [B, T, D] features. Старые
форматы [B], [B, T] и совместимый sequence_mask остаются поддержанными.

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

Batch.get_masks() возвращает все masks, а с аргументом — один flat key. Для
совместимости читаются также legacy nested masks из masks["mask"] и meta["masks"].
