# API data

Data subsystem превращает datasets в DataLoader containers для Trainer.
DataModule строит train/validation/test. Доступны два встроенных collate:
legacy MaskingCollate с key masking_collate и
MultimodalSequenceCollate с key multimodal_sequence_collate. Формы, masks и
YAML examples описаны в [Data](../user-guide/data.md).

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
