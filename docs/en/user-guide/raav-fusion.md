# RAAV fusion

RAAVFusionModel treats configured frame modalities as aligned temporal streams
and uses a separate context modality as a bottleneck. It first computes
per-frame, mask-aware modality weights, then lets trainable context latents
attend to the context sequence. Each frame attends to those context latents,
receives a context residual, and can optionally pass through a temporal module.

## Parameters

~~~python
RAAVFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    frame_modalities: tuple[str, ...],
    context_modality: str,
    num_latents: int = 4,
    gate_hidden: int = 64,
    temporal: nn.Module | None = None,
)
~~~

- input_dims declares the feature width of every frame and context modality;
  internal projections map them to shared_dim.
- shared_dim is the common attention width and output_dim is the prediction
  width.
- frame_modalities is a non-empty tuple of aligned temporal modality names.
- context_modality names the independent context stream.
- num_latents controls the context bottleneck and must be positive.
- gate_hidden is the hidden width of the frame-modality gate.
- temporal is an optional Python module. It must preserve [B, T, shared_dim];
  TransformerEncoder receives the frame padding mask.

## Inputs and outputs

Frame inputs have shape [B, T, D] after collation and must have the same T
across present frame modalities. The context input may have a different
temporal length. Predictions have shape [B, T, output_dim], with invalid frame
positions zeroed.

`<modality>_temporal_mask` is preferred, followed by `<modality>_mask` and a
compatible sequence_mask. Missing frame modalities are skipped, while
per-sample missing frames are removed by modality masks. A missing or fully
masked context produces zero context latents and zero context contribution.
An empty context sequence is supported. An empty frame sequence produces an
empty prediction sequence; at least one frame modality must be available.

Useful ModelOutput.aux values are frame_weights, frame_mask, fused, context,
context_mask, context_latent_attention, frame_to_context_attention,
context_delta, and refined.

## Python

~~~python
from chimera_ml.models import RAAVFusionModel

model = RAAVFusionModel(
    input_dims={"video": 512, "audio": 768, "text": 1024},
    shared_dim=256,
    output_dim=2,
    frame_modalities=("video", "audio"),
    context_modality="text",
    num_latents=4,
)
~~~

## YAML

~~~yaml
model:
  name: raav_fusion_model
  params:
    input_dims:
      video: 512
      audio: 768
      text: 1024
    shared_dim: 256
    output_dim: 2
    frame_modalities:
      - video
      - audio
    context_modality: text
    num_latents: 4
    gate_hidden: 64
~~~

The collate step does not align timelines automatically. Prepare frame streams
on a common timeline before collation, or validate their lengths in the dataset.

## Synthetic forward

Frame modalities must share their temporal length; the context timeline may differ:

```python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import RAAVFusionModel

model = RAAVFusionModel(
    {"video": 4, "audio": 5, "text": 6}, shared_dim=8, output_dim=2,
    frame_modalities=("video", "audio"), context_modality="text",
).eval()
batch = Batch(inputs={
    "video": torch.randn(2, 3, 4),
    "audio": torch.randn(2, 3, 5),
    "text": torch.randn(2, 5, 6),
}, targets=None)
output = model(batch)
assert output.preds.shape == (2, 3, 2)
assert output.aux["frame_mask"].shape == (2, 3)
```

Changing `audio` to a different frame length raises `ValueError`; a different `text` length is supported.
