# RAAV fusion

`raav_fusion_model` treats configured frame modalities as temporal streams,
learns per-frame modality weights, and refines the fused stream with latent
queries over an optional context modality. It returns `frame_weights`, `fused`,
`context`, and `context_mask` in `ModelOutput.aux`.
