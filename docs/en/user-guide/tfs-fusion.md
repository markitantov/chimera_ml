# TFS fusion

TFSFusionModel implements feature-only TripleFusion. It projects each configured
modality to shared_dim, computes directed cross-modal interactions, and produces
one prediction vector per sample. With exactly three modalities and default
groups it follows the MASAI-style hierarchy:

1. six independent directed Transformer cross-attention blocks;
2. two triples made from the directed pairs;
3. two reciprocal second-level cross-attention blocks;
4. masked mean and population-standard-deviation pooling;
5. Linear -> BatchNorm -> Tanh, followed by the internal prediction layer.

This is hierarchical fusion, not a simple average of pair outputs.

## Parameters

~~~python
TFSFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    groups: tuple[tuple[tuple[str, str], ...], ...] | None = None,
)
~~~

- input_dims maps each modality name to its input feature width. The model
  creates one internal projection per modality.
- shared_dim is the width used by all fusion blocks.
- output_dim is the width of ModelOutput.preds.
- groups is optional. By default every unordered pair contributes both
  directions. A custom grouping or a non-default modality count uses the
  generalized grouped hierarchy instead of the three-modality MASAI path.

## Inputs and outputs

Batch.inputs contains feature-only tensors with shape [B, D] or [B, T, D].
Vectors are normalized internally to one temporal position.
ModelOutput.preds has shape [B, output_dim].

Masks are read from Batch.masks, including legacy nested masks. A dedicated
`<modality>_temporal_mask` with shape [B, T] takes precedence; otherwise
`<modality>_mask` may have shape [B] or [B, T]. A compatible sequence_mask is also
combined. Invalid positions do not contribute to attention or pooling, and
invalid directed pairs are zero before second-level fusion. A modality can be
missing from Batch.inputs; per-sample absence is represented by a false mask.
At least one directed pair must be available.

The auxiliary output contains `pair_<source>-><target>`,
`pair_valid_<source>-><target>`, pair_weights, group_weights, and fused. The main
prediction is in ModelOutput.preds; aux is for diagnostics and feature use.

## Python

~~~python
from chimera_ml.models import TFSFusionModel

model = TFSFusionModel(
    input_dims={"text": 1024, "audio": 768, "video": 512},
    shared_dim=256,
    output_dim=2,
)
~~~

## YAML

~~~yaml
model:
  name: tfs_fusion_model
  params:
    input_dims:
      text: 1024
      audio: 768
      video: 512
    shared_dim: 256
    output_dim: 2
~~~

Use groups for explicit directed-pair groups. The YAML value is a list of
groups, where each pair is a two-item list such as [text, audio]. Empty groups
and repeated directed pairs are invalid. A three-modality input with a missing
whole tensor can take the generalized path; when all three tensors are present,
the default hierarchy is used.
