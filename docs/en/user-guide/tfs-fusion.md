# TFS fusion

`tfs_fusion_model` builds a shared representation for each configured
modality, supports `[B, D]` and temporal `[B, T, D]` inputs, and computes
directed attention pairs. Pair outputs are averaged within optional groups,
refined by a group-specific linear layer, averaged across groups, and passed
to the supplied head.

The model reads `<modality>_mask` from `Batch.masks` when available. Missing
modalities are skipped; at least two configured modalities must be present.
When `input_dims` contains a modality, its encoder output is projected to
`shared_dim` before fusion.
