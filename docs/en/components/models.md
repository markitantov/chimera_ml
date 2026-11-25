# Models

## Model interface

`BaseModel` is an abstract `torch.nn.Module` with `forward(batch: Batch) -> ModelOutput`. A model declares its accepted modality names and tensor shapes through its constructor or plugin contract, consumes `Batch.inputs` and masks, and returns task predictions in `ModelOutput.preds`. It should raise a clear `ValueError` for missing required modalities or incompatible widths.

Minimal registered model:

```python
import torch
from torch import nn
from chimera_ml.core import Batch, MODELS, ModelOutput
from chimera_ml.models import BaseModel

class MeanModel(BaseModel):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.head = nn.Linear(input_dim, output_dim)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch.inputs["features"]
        if x.ndim != 2:
            raise ValueError("features must have shape [B, D]")
        return ModelOutput(preds=self.head(x))

@MODELS.register("mean_model")
def mean_model(**params):
    return MeanModel(**params)
```

```yaml
model:
  name: mean_model
  params: {input_dim: 16, output_dim: 2}
```

Training and evaluation switch the module between `train()` and `eval()`; the model is responsible for honoring that mode. Inference prediction steps must also set compatible models to evaluation mode.

## Built-in architectures

| Key | Class | Principle | Output |
| --- | --- | --- | --- |
| `tfs_fusion_model` | `TFSFusionModel` | Directed pairwise cross-modal hierarchy with masked statistical pooling | `[B, output_dim]` |
| `tfs_dcmmmoe_fusion_model` | `TFSDCMMOEFusionModel` | TFS hierarchy plus learned masked pair gates | `[B, output_dim]` |
| `raav_fusion_model` | `RAAVFusionModel` | Aligned frame fusion with context latent bottleneck | `[B,T,output_dim]` |
| `tomar_fusion_model` | `TOMARFusionModel` | Required anchor plus gated residual corrections | `[B, output_dim]` |

All accept feature tensors `[B,D]` or `[B,T,D]` and project configured widths to `shared_dim`. TFS variants require at least two available modality tensors. RAAV requires at least one frame modality and equal frame lengths among present frame streams; context length may differ. TOMAR requires the anchor for every sample. Fully masked rows are handled by each architecture's documented mask path; absent tensors and masks are not interchangeable.

See the [fusion comparison](fusion.md) and the individual [TFS](../user-guide/tfs-fusion.md), [TFS-DCMMOE](../user-guide/tfs-dcmmmoe-fusion.md), [RAAV](../user-guide/raav-fusion.md), and [TOMAR](../user-guide/tomar-fusion.md) pages.
