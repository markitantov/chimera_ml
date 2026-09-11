# API data

Data subsystem превращает datasets в \`DataLoader\` containers, которые
потребляет \`Trainer\`. \`DataModule\` строит train/validation/test loaders;
\`MaskingCollate\` превращает sample mappings в typed \`Batch\`, дополняет
variable-length inputs padding и записывает presence/sequence masks. Практика:
[Data](../user-guide/data.md).

## Data module

::: chimera_ml.data.datamodule.DataModule

## Collation и masks

::: chimera_ml.data.masking_collate.MaskingCollate

Registry key default collator: \`masking_collate\`.

::: chimera_ml.data.masking_collate.masking_collate

## Loader utilities

Helpers нормализуют single loader, mapping или sequence в stable split names,
которые используют metrics, callbacks и artifact paths.

::: chimera_ml.data.loader_utils.normalize_loaders

::: chimera_ml.data.loader_utils.sanitize_split_name
