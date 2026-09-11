# Data API

The data subsystem turns datasets into the \`DataLoader\` containers consumed by
\`Trainer\`. \`DataModule\` handles common train/validation/test construction;
\`MaskingCollate\` converts sample mappings into a typed \`Batch\`, pads variable
length inputs, and records presence/sequence masks. See [Data](../user-guide/data.md)
for dataset and YAML examples.

## Data module

::: chimera_ml.data.datamodule.DataModule

## Collation and masks

::: chimera_ml.data.masking_collate.MaskingCollate

The registry key for the default collator is \`masking_collate\`.

::: chimera_ml.data.masking_collate.masking_collate

## Loader utilities

These helpers normalize a single loader, a mapping, or a sequence into stable
split names used by metrics, callbacks, and artifact paths.

::: chimera_ml.data.loader_utils.normalize_loaders

::: chimera_ml.data.loader_utils.sanitize_split_name
