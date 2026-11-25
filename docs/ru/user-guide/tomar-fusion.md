# Слияние TOMAR

`TOMARFusionModel` использует одну модальность как опорную (`anchor`) и добавляет остаточные поправки от остальных заданных модальностей. Для каждой поправки отдельные проекция и MLP вычисляют шлюз по конкатенации опорного представления и усреднённого представления дополнительной модальности. Опорное представление и управляемые поправки нормализуются перед внутренним слоем предсказания.

## Параметры

~~~python
TOMARFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    anchor_modality: str,
    output_dim: int,
    gate_hidden: int = 64,
)
~~~

`input_dims` задаёт положительные ширины признаков и внутренние проекции. `shared_dim` — ширина остаточного представления, `anchor_modality` — обязательная модальность, `output_dim` — ширина предсказания, `gate_hidden` — ширина MLP для шлюза каждой дополнительной модальности.

## Входы, маски и выходы

Входы — признаки формы `[B, D]` или `[B, T, D]`. Временные признаки маскируются и сворачиваются в `[B, shared_dim]`; предсказание имеет форму `[B, output_dim]`. Явные временные маски имеют приоритет над устаревшими масками наличия. Отсутствующая или полностью замаскированная дополнительная модальность не добавляет поправку. Тензор опорной модальности (`anchor`) должен существовать и быть доступен для каждого образца; иначе модель выбрасывает `ValueError`.

В `ModelOutput.aux` находятся опорное представление `anchor`, по одному коэффициенту `<rho_modality>` и остаточной поправке `<delta_modality>` для каждой доступной дополнительной модальности, а также объединённое представление `fused`. Основной результат находится в `preds`.

## Python API

~~~python
from chimera_ml.models import TOMARFusionModel

model = TOMARFusionModel(
    input_dims={"text": 1024, "audio": 768},
    shared_dim=256,
    anchor_modality="text",
    output_dim=2,
)
~~~

## Конфигурация YAML

~~~yaml
model:
  name: tomar_fusion_model
  params:
    input_dims:
      text: 1024
      audio: 768
    shared_dim: 256
    anchor_modality: text
    output_dim: 2
    gate_hidden: 64
~~~

Опорная модальность (`anchor`) обязательна. Другие модальности могут отсутствовать в `Batch.inputs`, быть недоступными для отдельных образцов или быть пустыми.

## Синтетический forward

Опорная модальность (`anchor`) должна присутствовать и быть валидной у каждого образца; не-опорные модальности можно не подавать:

```python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import TOMARFusionModel
model = TOMARFusionModel({"text": 4, "audio": 5}, shared_dim=8, anchor_modality="text", output_dim=2).eval()
batch = Batch(inputs={"text": torch.randn(2, 3, 4), "audio": torch.randn(2, 3, 5)}, targets=None)
output = model(batch)
assert output.preds.shape == (2, 2)
assert output.aux["fused"].shape == (2, 8)
```

Отсутствующая или полностью замаскированная опорная модальность (`anchor`) приводит к `ValueError`; отсутствующая не-опорная модальность не добавляет остаточную поправку.
