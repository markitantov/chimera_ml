# Data components

## Dataset interface

The core does not register a concrete dataset (`DATAMODULES` is empty in the built-in registry). A plugin dataset is normally a `torch.utils.data.Dataset` whose item is a mapping with `inputs`, optional `targets`/`target`, and optional `meta`/`masks`:

```python
{
    "inputs": {"audio": torch.randn(10, 80), "video": None},
    "targets": torch.tensor([0, 1]),
    "meta": {"sample_id": "s-1"},
}
```

The dataset owns file formats, feature extraction, labels, IDs, and task-specific missing-data policy. The collate owns batching, padding, and canonical masks.

## `DataModule`

`DataModule` accepts `train_dataset`, `val_dataset`, and `test_dataset`, each as one Dataset, a mapping of named Datasets, a sequence of Datasets, or `None`. Constructor defaults are `batch_size=32`, `num_workers=0`, `pin_memory=True`, `persistent_workers=False`, `shuffle_train=True`, `drop_last_train=False`, and `collate_fn=MaskingCollate()`.

```python
from chimera_ml.data import DataModule, multimodal_sequence_collate

module = DataModule(
    train_dataset=train_dataset,
    val_dataset={"clean": val_clean, "noisy": val_noisy},
    test_dataset=test_dataset,
    batch_size=16,
    num_workers=2,
    pin_memory=True,
    collate_fn=multimodal_sequence_collate(pad_sequences=True),
)
train_loader = module.train_dataloader()
val_loaders = module.val_dataloader()  # mapping is preserved
```

`train_dataloader`, `val_dataloader`, and `test_dataloader` return a DataLoader, a mapping of DataLoaders, a sequence, or `None`, preserving the dataset container shape. Training shuffling/drop-last apply only to the train split. A plugin DataModule factory must accept `collate_fn` if the YAML `collate` section is used; configuring both top-level `collate` and `data.params.collate_fn` is rejected.

## Registered collates

| Key | Best fit | Padding and masks |
| --- | --- | --- |
| `masking_collate` | General/legacy samples and a shared sequence notion | Pads first tensor dimension to a shared maximum; emits `sequence_mask` and `<modality>_mask`. |
| `multimodal_sequence_collate` | Independent modality timelines | Pads each modality independently; emits `<modality>_mask` and `<modality>_temporal_mask`. |

### `masking_collate`

Factory parameters are `pad_sequences=True`, `include_legacy_meta_masks=False`, and `pad_value=0.0`. It accepts non-empty sample mappings. Present tensors must have equal rank and equal feature dimensions after axis 0. With `pad_sequences=True`, `[T,D]` samples become `[B,T_max,D]`; with `False`, all present shapes must match. Missing rows are filled with `pad_value`, presence is a float32 `[B]` tensor, and `sequence_mask` is boolean `[B,T_max]`. Targets accept either `target` or `targets` and are padded using the same first-dimension rule; if all are absent, `Batch.targets` is `None`. Empty input raises `ValueError`; incompatible ranks/dimensions raise `ValueError`.

```python
collate = masking_collate(pad_sequences=True, pad_value=-1.0)
batch = collate([
    {"inputs": {"audio": torch.ones(3, 4)}, "targets": torch.tensor([1.0])},
    {"inputs": {"audio": torch.ones(1, 4)}, "targets": torch.tensor([2.0])},
])
# audio [2, 3, 4], audio_mask [2], sequence_mask [2, 3]
```

### `multimodal_sequence_collate`

Factory parameters are the same three. Rank-one `[D]` inputs are stacked as `[B,D]` with temporal mask `[B,1]`; rank-two `[T,D]` inputs are padded independently to `[B,T_max,D]`. A missing modality row is zero-filled, `<modality>_mask` is float32 `[B]`, and `<modality>_temporal_mask` is boolean `[B,T_max]`. Explicit temporal masks may be supplied under sample `meta["<modality>_temporal_mask"]`, `meta["masks"]`, or sample-level `masks`; their lengths must match the present sequence. Ranks other than 1/2, incompatible feature widths, invalid explicit masks, empty batches, and unequal lengths with `pad_sequences=False` raise `ValueError`.

```python
collate = multimodal_sequence_collate()
batch = collate([
    {"inputs": {"audio": torch.ones(2, 4), "video": torch.ones(3, 6)}},
    {"inputs": {"audio": torch.ones(1, 4)}},
])
# audio [2,2,4], audio_mask [1,1], audio_temporal_mask [2,2]
# video [2,3,6], video_mask [1,0], video_temporal_mask [2,3]
```

These collates do not align audio and video timelines. The dataset must resample or otherwise establish aligned frame timelines before using architectures such as RAAV.
