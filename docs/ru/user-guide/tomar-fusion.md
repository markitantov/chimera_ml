# TOMAR fusion

TOMARFusionModel использует одну модальность как anchor и добавляет остаточные
поправки от остальных заданных модальностей. Для каждой поправки отдельные
проекция и MLP вычисляют шлюз по конкатенации anchor и усреднённого
представления дополнительной модальности. Anchor и управляемые поправки
нормализуются перед внутренним слоем предсказания.

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

input_dims задаёт положительные ширины признаков и внутренние проекции.
shared_dim — ширина остаточного представления, anchor_modality —
обязательная модальность, output_dim — ширина предсказания, gate_hidden —
ширина MLP для шлюза каждой дополнительной модальности.

## Входы, маски и выходы

Входы — признаки формы [B, D] или [B, T, D]. Временные признаки
маскируются и сворачиваются в [B, shared_dim]; предсказание имеет форму
[B, output_dim]. Выделенные временные маски имеют приоритет над legacy-масками
присутствия. Отсутствующая или полностью замаскированная дополнительная
модальность не добавляет поправку. Тензор anchor должен существовать и быть
доступен для каждого образца; иначе модель выбрасывает ValueError.

В ModelOutput.aux находятся anchor, по одному коэффициенту `<rho_modality>` и
остаточной поправке `<delta_modality>` для каждой доступной дополнительной
модальности, а также fused. Основной результат находится в preds.

## Python

~~~python
from chimera_ml.models import TOMARFusionModel

model = TOMARFusionModel(
    input_dims={"text": 1024, "audio": 768},
    shared_dim=256,
    anchor_modality="text",
    output_dim=2,
)
~~~

## YAML

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

Anchor является обязательным. Другие модальности могут отсутствовать в
Batch.inputs, быть недоступными для отдельных образцов или быть пустыми.
