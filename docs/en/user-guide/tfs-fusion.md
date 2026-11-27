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

## Theory-aligned configuration

Set `theory_aligned=True` to implement UDKMO equations (1.4)-(1.5). Every
ordered modality pair must occur exactly once in `groups`. Directed pair
attention coefficients are retained as `pair_attention_<source>-><target>`;
inter-group coefficients are retained as `group_attention_<index>`, where the
group index identifies the query group. The temporal interaction sequences are
kept until this attention is complete. Each available group is then masked-mean
pooled and the resulting group vectors are averaged with equal weights. This is
not availability weighting. The default branch remains a backward-compatible
legacy TFS mode and is not an exact implementation of equation (1.5).

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
invalid directed pairs are zero before second-level fusion. A directed pair
can be formed only when at least two configured modality tensors are present in Batch.inputs. Their masks may still mark every temporal
position invalid for a sample. In that case pair_valid and pair_weights are zero,
fused is zero, and the internal prediction layer still returns preds. If fewer
than two modality tensors are present, no pair can be formed and the model raises
ValueError.

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

## Synthetic forward

The smallest useful check uses two or more configured modality tensors. The three-modality example below returns one prediction vector per sample and exposes pair diagnostics:

```python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import TFSFusionModel

model = TFSFusionModel({"a": 4, "b": 5, "c": 6}, shared_dim=8, output_dim=2).eval()
batch = Batch(inputs={"a": torch.randn(2, 3, 4), "b": torch.randn(2, 3, 5), "c": torch.randn(2, 3, 6)}, targets=None)
output = model(batch)
assert output.preds.shape == (2, 2)
assert "pair_weights" in output.aux
```

`groups` and modality masks must describe the same configured names. A missing whole tensor can use the generalized path, but fewer than two available tensors raises `ValueError`.
