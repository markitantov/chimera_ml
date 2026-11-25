# Fusion architecture comparison

These are the four built-in registered models. Their `aux` values are architecture diagnostics, not XAI modality influence scores.

| Architecture | Required inputs | Temporal behavior | Missing data | Notable `aux` |
| --- | --- | --- | --- | --- |
| TFS | At least two configured tensors per forward | Accepts `[B,D]` or `[B,T,D]`, pools to sample output | Missing whole tensors are skipped; fully masked pairs become zero | Pair values/validity, `pair_weights`, `group_weights`, `fused` |
| TFS-DCMMOE | Same as TFS | Same hierarchy and shapes | Masked pair experts receive zero adaptive weight | TFS diagnostics plus learned pair weights |
| RAAV | Non-empty `frame_modalities`; context is configured separately | Frame modalities must share `T`; context `T` may differ; output remains per frame | Missing frame streams are skipped; missing context gives zero context contribution | `frame_weights`, `frame_mask`, context/latent attention, `refined` |
| TOMAR | Anchor tensor available for every sample | Pools temporal inputs; sample output | Missing non-anchor contributes no residual; missing anchor raises | `anchor`, `rho_<modality>`, `delta_<modality>`, `fused` |

## TFS and TFS-DCMMOE

TFS uses directed pair interactions and a hierarchical aggregation. With exactly three names and default groups it uses six directed Transformer cross-attention blocks, reciprocal second-level blocks, masked mean/std pooling, and a dense/batchnorm/tanh projection. TFS-DCMMOE adds an input-dependent gate over available pair experts. `groups` is an optional nested tuple/list of directed pairs; invalid, repeated, or empty groups raise `ValueError`.

Both accept `input_dims`, `shared_dim`, `output_dim`, and optional `groups`; DCMMOE also has `gate_hidden=64`. They return `[B, output_dim]`. If fewer than two configured tensors exist in `Batch.inputs`, they raise; if all positions are masked, fused/pair weights are zero but the prediction head still returns a tensor.

## RAAV

RAAV first gates aligned frame modalities at each time, compresses the context modality into `num_latents=4` learned latents, attends frames to the context, and optionally applies `temporal`. Parameters are `input_dims`, `shared_dim`, `output_dim`, `frame_modalities`, `context_modality`, `num_latents=4`, `gate_hidden=64`, and `temporal=None`. `temporal` must preserve `[B,T,shared_dim]`; `TransformerEncoder` receives frame padding masks.

## TOMAR

TOMAR pools a required `anchor_modality`, computes a gated residual for each other configured modality, normalizes anchor plus corrections, and returns `[B, output_dim]`. Parameters are `input_dims`, `shared_dim`, `anchor_modality`, `output_dim`, and `gate_hidden=64`. The anchor must be present and valid for every sample.

Constructor examples and YAML are kept on each individual architecture page; use them rather than inventing a generic “fusion” parameter set.
