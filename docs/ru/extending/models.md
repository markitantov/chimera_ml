# Пользовательские модели

Наследуйте `BaseModel`, принимайте параметры конструктора, совместимые с YAML, читайте `Batch`, возвращайте `ModelOutput` и явно проверяйте отсутствие модальности и ширину признаков.

```python
import torch
from torch import nn
from chimera_ml.core import Batch, MODELS, ModelOutput
from chimera_ml.models import BaseModel

class LinearFeaturesModel(BaseModel):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__(); self.input_dim = input_dim; self.head = nn.Linear(input_dim, output_dim)
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

В YAML указывается ключ реестра, а не имя класса Python. Дополнительные тензоры можно поместить в `ModelOutput.aux`; функция потерь и метрики Trainer используют `preds`, если пользовательский компонент специально не читает `aux`.
