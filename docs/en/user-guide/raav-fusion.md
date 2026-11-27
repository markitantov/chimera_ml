# RAAV fusion

RAAVFusionModel performs frame-level fusion with contextual refinement. Frame
modalities are combined at every time position using learned reliability
weights. Each context modality has its own learnable bottleneck latents. A
compression attention block updates those latents from the context sequence;
frame attention then uses them as keys and values to produce a correction for
every frame. The result remains a sequence.

## Constructor

~~~python
RAAVFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    frame_modalities: Sequence[str],
    context_modalities: Sequence[str],
    num_context_latents: int = 4,
    gate_hidden: int | None = None,
    dropout: float = 0.1,
    num_heads: int = 4,
)
~~~

frame_modalities and context_modalities must be non-empty and disjoint.
num_context_latents must be positive and is applied independently to every
context modality. Context streams may have different lengths from frame
streams. Frame streams must share the same temporal length.

## Inputs and outputs

Frame inputs have shape [B, T, D]; two-dimensional inputs are treated as one
time position. Predictions have shape [B, T, output_dim]. Presence, temporal,
and sequence masks determine valid positions. Frame reliability coefficients
are returned as aux["frame_weights"].

Other useful auxiliary values are frame_mask, fused, context_delta,
context_<name>, context_mask_<name>, context_attention_<name>, and
context_compression_attention_<name>. Missing or fully masked context
contributes zero and produces no attention weights. At least one frame modality
must be present. Checkpoints from the previous pooled-context implementation
are not compatible with the learnable-latent parameter structure.

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
