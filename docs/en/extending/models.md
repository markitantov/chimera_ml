# Custom models

Subclass `BaseModel`, accept constructor parameters that can be represented in YAML, consume `Batch`, and return `ModelOutput`. Validate required modalities and widths explicitly.

```python
import torch
from torch import nn
from chimera_ml.core import Batch, MODELS, ModelOutput
from chimera_ml.models import BaseModel

class LinearFeaturesModel(BaseModel):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.head = nn.Linear(input_dim, output_dim)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch.inputs.get("features")
        if x is None or x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError("features must have shape [B, configured_input_dim]")
        return ModelOutput(preds=self.head(x))

@MODELS.register("linear_features_model")
def linear_features_model(input_dim: int, output_dim: int = 1):
    return LinearFeaturesModel(input_dim, output_dim)
```

```yaml
model:
  name: linear_features_model
  params: {input_dim: 16, output_dim: 2}
```

The registry key, not the Python class name, is used in YAML. Auxiliary tensors may be placed in `ModelOutput.aux`; Trainer loss/metrics use `preds` unless a custom component explicitly consumes aux.
