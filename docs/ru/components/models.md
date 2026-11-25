# Модели

## Интерфейс модели

`BaseModel` — abstract `torch.nn.Module` с `forward(batch: Batch) -> ModelOutput`. Модель описывает допустимые modality names и widths, читает `Batch.inputs` и masks и возвращает task predictions в `ModelOutput.preds`. При missing required modality или incompatible width должна выдавать понятный `ValueError`.

```python
import torch
from torch import nn
from chimera_ml.core import Batch, MODELS, ModelOutput
from chimera_ml.models import BaseModel

class MeanModel(BaseModel):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__(); self.head = nn.Linear(input_dim, output_dim)
    def forward(self, batch: Batch) -> ModelOutput:
        x = batch.inputs["features"]
        if x.ndim != 2:
            raise ValueError("features must have shape [B, D]")
        return ModelOutput(preds=self.head(x))

@MODELS.register("mean_model")
def mean_model(**params): return MeanModel(**params)
```

```yaml
model:
  name: mean_model
  params: {input_dim: 16, output_dim: 2}
```

Trainer переключает `train()`/`eval()`, а prediction step должен сам установить совместимую модель в `eval()`.

## Встроенные архитектуры

| Key | Class | Principle | Output |
| --- | --- | --- | --- |
| `tfs_fusion_model` | `TFSFusionModel` | Directed pairwise hierarchy и masked statistics | `[B, output_dim]` |
| `tfs_dcmmmoe_fusion_model` | `TFSDCMMOEFusionModel` | TFS плюс learned masked pair gates | `[B, output_dim]` |
| `raav_fusion_model` | `RAAVFusionModel` | Aligned frame fusion и context latents | `[B,T,output_dim]` |
| `tomar_fusion_model` | `TOMARFusionModel` | Required anchor и gated residuals | `[B, output_dim]` |

Все принимают feature tensors `[B,D]` или `[B,T,D]` и проецируют widths в `shared_dim`. TFS variants требуют минимум два доступных modality tensors; RAAV — хотя бы одну frame modality и одинаковый `T` среди frame streams; TOMAR требует anchor для каждого sample.

См. [сравнение fusion](fusion.md) и страницы [TFS](../user-guide/tfs-fusion.md), [TFS-DCMMOE](../user-guide/tfs-dcmmmoe-fusion.md), [RAAV](../user-guide/raav-fusion.md), [TOMAR](../user-guide/tomar-fusion.md).
