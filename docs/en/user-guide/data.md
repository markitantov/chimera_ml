# Preparing datasets and loaders

## Dataset responsibility

A task dataset is a PyTorch `Dataset` (or a plugin equivalent) whose `__getitem__` returns a mapping. The framework collates these mappings; it does not prescribe how audio, video, text, cached features, labels, or IDs are stored.

A practical sample has this shape:

```python
{
    "inputs": {
        "audio": torch.Tensor,  # [T_audio, D_audio] or [D_audio]
        "video": torch.Tensor,  # [T_video, D_video] or [D_video]
    },
    "targets": torch.Tensor,     # e.g. [2] or [T_target, 2]
    "meta": {"id": "sample-001"},
}
```

`target` is accepted as an alias for `targets`. Inputs may omit a modality for a sample. Unlabeled samples can set targets to `None`, but `Trainer.fit` rejects a training batch without targets.

## DataModule and splits

The framework `DataModule` accepts one dataset, a mapping of named datasets, or a sequence for each split. It creates DataLoaders with `batch_size`, `num_workers`, `pin_memory`, `persistent_workers`, `shuffle_train`, and `drop_last_train`:

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

A plugin DataModule factory normally constructs these datasets and returns the same interface. The current core does not register a generic datamodule key, so the `data.name` in a training YAML must come from your plugin.

A split mapping preserves names. A sequence becomes names such as `train0`; evaluation normalizes names and prefixes unnamed split loaders with `train`, `val`, or `test`. Multiple train loaders are scheduled by `train_loader_mode` in `TrainConfig`.

## Batch and canonical masks

`Batch.inputs` maps modality names to tensors. `Batch.targets` is the task target or `None`. `Batch.masks` is the canonical flat mask mapping; `Batch.get_masks()` also reads legacy nested masks from `masks["mask"]` or `meta["masks"]`.

The built-in collates use these semantics:

- `<modality>_mask` is sample presence. The built-in multimodal collate emits a float32 `[B]` tensor; a zero means that sample has no usable modality tensor.
- `<modality>_temporal_mask` is valid temporal positions, boolean `[B, T]`.
- `sequence_mask` from `masking_collate` marks valid padded sequence positions, boolean `[B, T_max]`.
- Missing inputs are zero-filled by the collate, but the presence/temporal masks are what models use to distinguish missing data from real zeros.

## Choosing a collate

### `masking_collate`

Use `masking_collate` for the legacy/general behavior. It accepts tensor fields, pads the first tensor dimension when `pad_sequences: true`, emits `sequence_mask`, and emits one `<modality>_mask` per modality. `pad_sequences: false` requires equal shapes for each present field.

```yaml
collate:
  name: masking_collate
  params:
    pad_sequences: true
    pad_value: 0.0
    include_legacy_meta_masks: false
```

A batch of `audio` sequences with lengths 3 and 1 produces `audio.shape == [2, 3, D]`, `audio_mask.shape == [2]`, and `sequence_mask.shape == [2, 3]`.

### `multimodal_sequence_collate`

Use this for independent feature timelines. Rank-one samples `[D]` become `[B, D]` with temporal mask `[B, 1]`; rank-two samples `[T, D]` are padded independently per modality to `[B, T_max, D]`. Audio and video do not share a temporal maximum.

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

An explicit per-sample `<modality>_temporal_mask` may be supplied under `meta`, `meta["masks"]`, or a sample-level `masks` mapping. Its length must equal the present sequence length. `pad_value`, `pad_sequences`, and `include_legacy_meta_masks` are supported parameters. Timelines are not aligned automatically.

Configure a collate at the top-level `collate` section for CLI `train` and `eval`. The selected DataModule factory must explicitly accept a keyword `collate_fn`; configuring both `collate:` and `data.params.collate_fn` is rejected.
