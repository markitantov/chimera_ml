# TFS fusion

TFSFusionModel implements hierarchical bidirectional fusion. Each configured
modality is projected to shared_dim. Every configured ordered pair is processed
with cross-modal attention. Pair results are grouped, refined by group-level
attention, pooled with a masked mean and population standard deviation, and
passed through Linear, BatchNorm, and Tanh before prediction.

## Constructor

~~~python
TFSFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    groups: Sequence[Sequence[Sequence[str]]] | None = None,
    dropout: float = 0.1,
    max_position_length: int = 5000,
)
~~~

groups defaults to two groups containing both directions of every unordered
pair. Each directed pair must occur once. Empty groups, unknown modalities,
self-pairs, and repeated pairs raise ValueError.

## Inputs and outputs

Inputs are feature tensors with shape [B, D] or [B, T, D]. The result has
shape [B, output_dim]. Temporal and presence masks are read from Batch; a
temporal mask has priority over a presence mask. sequence_mask is combined
when it has the same shape.

The auxiliary output contains pair representations, pair validity masks, pair
attention, group attention, group_weights, and fused. These values are model
diagnostics and are also available to the XAI step.

At least two configured modality tensors must be present in a batch. A fully
masked pair is retained as an invalid zero contribution.

## Example

~~~python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import TFSFusionModel

model = TFSFusionModel(
    {"text": 4, "audio": 5, "video": 6},
    shared_dim=8,
    output_dim=2,
).eval()

batch = Batch(
    inputs={
        "text": torch.randn(2, 3, 4),
        "audio": torch.randn(2, 3, 5),
        "video": torch.randn(2, 3, 6),
    },
    targets=None,
)
output = model(batch)
assert output.preds.shape == (2, 2)
~~~

Registry key: tfs_fusion_model.
