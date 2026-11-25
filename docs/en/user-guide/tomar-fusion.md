# TOMAR fusion

TOMARFusionModel treats one modality as an anchor and adds residual corrections
from the other configured modalities. Each correction is produced by a
modality-specific projection and is gated by an MLP conditioned on the
concatenation of the anchor and that modality's pooled representation. The
anchor plus all gated corrections is normalized before the internal prediction
layer.

## Parameters

~~~python
TOMARFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    anchor_modality: str,
    output_dim: int,
    gate_hidden: int = 64,
)
~~~

input_dims declares positive feature widths and creates internal projections.
shared_dim is the residual width, anchor_modality is required, output_dim is
the prediction width, and gate_hidden controls each extra modality's gate MLP.

## Inputs, masks, and outputs

Inputs are feature-only [B, D] or [B, T, D] tensors. Temporal inputs are
masked and pooled to [B, shared_dim]; predictions have shape [B, output_dim].
Dedicated temporal masks take precedence over legacy presence masks. A missing
or fully masked non-anchor modality contributes no residual. The anchor tensor
must exist and be available for every sample; otherwise the model raises
ValueError.

ModelOutput.aux contains anchor, one rho_`<modality>` coefficient and
delta_`<modality>` residual per available non-anchor modality, and fused. preds is
the task output.

## Python

~~~python
from chimera_ml.models import TOMARFusionModel

model = TOMARFusionModel(
    input_dims={"text": 1024, "audio": 768},
    shared_dim=256,
    anchor_modality="text",
    output_dim=2,
)
~~~

## YAML

~~~yaml
model:
  name: tomar_fusion_model
  params:
    input_dims:
      text: 1024
      audio: 768
    shared_dim: 256
    anchor_modality: text
    output_dim: 2
    gate_hidden: 64
~~~

The anchor is a hard requirement. Other modalities may be absent from
Batch.inputs, unavailable for individual samples, or empty.

## Synthetic forward

The anchor must be present and valid for every sample; other modalities may be omitted:

```python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import TOMARFusionModel

model = TOMARFusionModel({"text": 4, "audio": 5}, shared_dim=8, anchor_modality="text", output_dim=2).eval()
batch = Batch(inputs={"text": torch.randn(2, 3, 4), "audio": torch.randn(2, 3, 5)}, targets=None)
output = model(batch)
assert output.preds.shape == (2, 2)
assert output.aux["fused"].shape == (2, 8)
```

If the anchor is absent or fully masked for one sample, the forward raises `ValueError`; a missing non-anchor contributes no residual.
