# Модели

## Интерфейс модели

Модель преобразует входы `Batch` в предсказания `ModelOutput`. Она отвечает за вычисление представления и результата задачи, но получает данные и маски через установленный контракт пакета.

`BaseModel` — абстрактный `torch.nn.Module` с методом `forward(batch: Batch) -> ModelOutput`. Модель описывает допустимые имена и ширины модальностей, читает `Batch.inputs` и маски и возвращает предсказания задачи в `ModelOutput.preds`. При отсутствии обязательной модальности или несовместимой ширине она должна выдавать понятный `ValueError`.

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

Trainer переключает режимы `train()` и `eval()`, а шаг предсказания должен сам перевести совместимую модель в режим `eval()`.

## Встроенные архитектуры

| Ключ | Класс | Принцип работы | Результат |
| --- | --- | --- | --- |
| `tfs_fusion_model` | `TFSFusionModel` | Иерархия направленных пар и статистики с учётом масок | `[B, output_dim]` |
| `tfs_dcmmmoe_fusion_model` | `TFSDCMMOEFusionModel` | TFS и обучаемые шлюзы пар с учётом масок | `[B, output_dim]` |
| `raav_fusion_model` | `RAAVFusionModel` | Слияние выровненных кадров и латентные векторы контекста | `[B,T,output_dim]` |
| `tomar_fusion_model` | `TOMARFusionModel` | Обязательная опорная модальность и остаточные поправки со шлюзами | `[B, output_dim]` |

Все модели принимают тензоры признаков `[B,D]` или `[B,T,D]` и проецируют ширины признаков в `shared_dim`. Варианты TFS требуют как минимум два доступных тензора модальностей; RAAV — хотя бы одну кадровую модальность и одинаковый `T` у кадровых потоков; TOMAR требует опорную модальность (`anchor`) для каждого образца.

См. [сравнение fusion](fusion.md) и страницы [TFS](../user-guide/tfs-fusion.md), [TFS-DCMMOE](../user-guide/tfs-dcmmmoe-fusion.md), [RAAV](../user-guide/raav-fusion.md), [TOMAR](../user-guide/tomar-fusion.md).
