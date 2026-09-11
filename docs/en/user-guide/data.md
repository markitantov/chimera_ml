# Data and loaders

A datamodule factory returns an object with train_dataloader,
val_dataloader, and test_dataloader. The built-in DataModule creates PyTorch
DataLoader instances from a single dataset, a mapping of named datasets, or a
sequence of datasets.

The default MaskingCollate collects tensor fields into a Batch, pads
variable-length sequences when configured, and produces flat mask keys such as
audio_mask and sequence_mask. A plugin can register a custom collate under the
collates registry.

## Multiple loaders

A loader may be a single DataLoader, a mapping, or a list/tuple. Names are
normalized for stable metric keys. Training supports:

- single: consume the first loader;
- round_robin: alternate loaders in declaration order;
- weighted: sample loaders with train_loader_weights.

train_stop_on: min stops when the first active loader is exhausted; max
continues until all active loaders are exhausted. For weighted, a finite loader
can be removed in max mode. Mapping split names are sanitized to alphanumerics,
dot, underscore, and dash.

## Batch contract

A dataset sample should provide input tensors and may provide targets, masks,
and metadata.

~~~python
Batch(
    inputs={"audio": audio_tensor, "video": video_tensor},
    targets=target_tensor,
    masks={"audio_mask": audio_mask},
    meta={"id": sample_id},
)
~~~

Batch.get_masks accepts no argument for all masks or a specific flat key. For
compatibility it also reads legacy nested masks from masks["mask"] and
meta["masks"], flattening modality names to modality_mask.
