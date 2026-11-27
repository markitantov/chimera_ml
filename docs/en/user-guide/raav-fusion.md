# RAAV fusion

RAAVFusionModel performs frame-level fusion with contextual refinement. Frame
modalities are combined at every time position using learned reliability
weights. Each configured context modality is pooled, projected, and used to
produce a cross-modal correction for every frame. The result remains a
sequence.

## Constructor

~~~python
RAAVFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    frame_modalities: Sequence[str],
    context_modalities: Sequence[str],
    gate_hidden: int | None = None,
    dropout: float = 0.1,
    num_heads: int = 4,
)
~~~

frame_modalities and context_modalities must be non-empty and disjoint.
gate_hidden=None uses shared_dim. Context streams may have different lengths
from frame streams. Frame streams must share the same temporal length.

## Inputs and outputs

Frame inputs have shape [B, T, D]; two-dimensional inputs are treated as one
time position. Predictions have shape [B, T, output_dim]. Presence, temporal,
and sequence masks determine valid positions. Frame reliability coefficients
are returned as aux["frame_weights"].

Other useful auxiliary values are frame_mask, fused, context_delta,
context_<name>, context_mask_<name>, and context_attention_<name>. Missing
context contributes zero. At least one frame modality must be present.

## Example

~~~python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import RAAVFusionModel

model = RAAVFusionModel(
    {"video": 4, "audio": 5, "text": 6},
    shared_dim=8,
    output_dim=2,
    frame_modalities=("video", "audio"),
    context_modalities=("text",),
).eval()

batch = Batch(
    inputs={
        "video": torch.randn(2, 3, 4),
        "audio": torch.randn(2, 3, 5),
        "text": torch.randn(2, 5, 6),
    },
    targets=None,
)
output = model(batch)
assert output.preds.shape == (2, 3, 2)
~~~

Registry key: raav_fusion_model.
