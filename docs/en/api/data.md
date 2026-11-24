# Data API

The data subsystem turns datasets into DataLoader containers consumed by
Trainer. DataModule handles common train/validation/test construction. Two
built-in collates are available: the legacy MaskingCollate under
masking_collate and MultimodalSequenceCollate under
multimodal_sequence_collate. See [Data](../user-guide/data.md) for shapes,
masks, and YAML examples.

## Data module

::: chimera_ml.data.datamodule.DataModule

## Legacy collation

::: chimera_ml.data.masking_collate.MaskingCollate

::: chimera_ml.data.masking_collate.masking_collate

## Independent sequence collation

::: chimera_ml.data.multimodal_sequence_collate.MultimodalSequenceCollate

::: chimera_ml.data.multimodal_sequence_collate.multimodal_sequence_collate

## Loader utilities

::: chimera_ml.data.loader_utils.normalize_loaders

::: chimera_ml.data.loader_utils.sanitize_split_name
