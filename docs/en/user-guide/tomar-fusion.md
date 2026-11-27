# TOMAR fusion

TOMARFusionModel uses one modality as the anchor and adds residual corrections
from the other configured modalities. Every correction is produced by a
modality-specific multilayer perceptron. A second multilayer perceptron
computes a sigmoid coefficient from the anchor and the additional modality.
The normalized sum is passed to the prediction layer.

## Constructor

~~~python
TOMARFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    anchor_modality: str,
    output_dim: int,
    gate_hidden: int | None = None,
)
~~~

The anchor must be listed in input_dims. gate_hidden=None uses shared_dim.
The other configured modalities are optional at inference time.

## Inputs and outputs

Inputs have shape [B, D] or [B, T, D]. Temporal inputs are masked and pooled
to [B, shared_dim]. The result has shape [B, output_dim]. The anchor must be
present and valid for every sample.

ModelOutput.aux contains anchor, fused, delta_<name>, and rho_<name> for each
available additional modality. These coefficients and corrections are retained
for XAI diagnostics.

## Example

~~~python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import TOMARFusionModel

model = TOMARFusionModel(
    {"text": 4, "audio": 5},
    shared_dim=8,
    anchor_modality="text",
    output_dim=2,
).eval()

batch = Batch(
    inputs={
        "text": torch.randn(2, 3, 4),
        "audio": torch.randn(2, 3, 5),
    },
    targets=None,
)
output = model(batch)
assert output.preds.shape == (2, 2)
~~~

Registry key: tomar_fusion_model.
