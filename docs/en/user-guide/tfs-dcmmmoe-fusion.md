# Adaptive TFS-DCMMOE fusion

TFSDCMMOEFusionModel keeps the hierarchical TFS structure and adds
input-dependent gates over directed pair experts. Each available pair receives
a score from its pair representation; masked softmax removes unavailable pairs
before the weights are applied. The model can therefore select relations for
each sample instead of treating every available pair equally.

For three modalities and default groups, the pair experts are the same six
directed first-level Transformer blocks and two reciprocal second-level blocks
used by TFS. Ordinary TFS uses binary availability at this stage; DCMMOE uses
learned normalized pair weights while preserving zero weight for unavailable
pairs.

## Parameters

~~~python
TFSDCMMOEFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    groups: tuple[tuple[tuple[str, str], ...], ...] | None = None,
    gate_hidden: int = 64,
)
~~~

input_dims, shared_dim, output_dim, and groups have the same meaning as in TFS.
gate_hidden is the hidden width of the pair-gating MLP. All dimensions must be
positive.

## Inputs, masks, and outputs

Inputs are feature-only [B, D] or [B, T, D] tensors in Batch.inputs; the model
projects them to shared_dim. Predictions have shape [B, output_dim]. Dedicated
temporal masks are preferred over legacy presence masks, and fully masked pairs
receive zero weight. A directed pair can be formed only when at least two configured modality
tensors are present in Batch.inputs. Their masks may still mark every temporal
position invalid for a sample. In that case pair_valid and pair_weights are zero,
fused is zero, and the internal prediction layer still returns preds. If fewer
than two modality tensors are present, no pair can be formed and the model raises
ValueError.

ModelOutput.aux includes pair representations and validity keys, plus
pair_weights, group_weights, and fused. pair_weights exposes mask-aware
adaptive expert weights; use preds for the task output.

## Python

~~~python
from chimera_ml.models import TFSDCMMOEFusionModel

model = TFSDCMMOEFusionModel(
    input_dims={"text": 1024, "audio": 768, "video": 512},
    shared_dim=256,
    output_dim=2,
    gate_hidden=128,
)
~~~

## YAML

~~~yaml
model:
  name: tfs_dcmmmoe_fusion_model
  params:
    input_dims:
      text: 1024
      audio: 768
      video: 512
    shared_dim: 256
    output_dim: 2
    gate_hidden: 128
~~~

Use groups for an explicit grouped hierarchy. A non-default grouping or a
configuration with fewer than three modality names uses the generalized TFS
path while retaining adaptive masked pair weighting.

## Synthetic forward

```python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import TFSDCMMOEFusionModel

model = TFSDCMMOEFusionModel({"a": 4, "b": 5, "c": 6}, shared_dim=8, output_dim=2, gate_hidden=16).eval()
batch = Batch(inputs={"a": torch.randn(2, 3, 4), "b": torch.randn(2, 3, 5), "c": torch.randn(2, 3, 6)}, targets=None)
output = model(batch)
assert output.preds.shape == (2, 2)
assert output.aux["pair_weights"].shape[0] == 2
```

The adaptive gate changes pair weights per sample; it does not make missing tensors available. Fewer than two available modality tensors raises `ValueError`.
