# Слияние TOMAR

TOMARFusionModel использует одну модальность как опорную и добавляет
остаточные поправки от остальных настроенных модальностей. Каждая поправка
вычисляется отдельным многослойным перцептроном. Второй многослойный
перцептрон получает опорное и дополнительное представления и вычисляет
сигмоидальный коэффициент. Нормированная сумма передаётся в слой
предсказания.

## Конструктор

~~~python
TOMARFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    anchor_modality: str,
    output_dim: int,
    gate_hidden: int | None = None,
)
~~~

Опорная модальность должна быть указана в input_dims. При gate_hidden=None
используется shared_dim. Остальные настроенные модальности могут отсутствовать
при выполнении модели.

## Входы и выходы

Входы имеют форму [B, D] или [B, T, D]. Временные признаки маскируются и
усредняются до [B, shared_dim]. Результат имеет форму [B, output_dim].
Опорная модальность должна присутствовать и быть доступной для каждого образца.

В ModelOutput.aux находятся anchor, fused, а также delta_<name> и rho_<name>
для каждой доступной дополнительной модальности. Эти коэффициенты и поправки
сохраняются для диагностики шага XAI.

## Пример

~~~python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import TOMARFusionModel

model = TOMARFusionModel(
    {"text": 4, "audio": 5},
    shared_dim=8,
    anchor_modality="text",
    output_dim=2,
).eval()

batch = Batch(
    inputs={
        "text": torch.randn(2, 3, 4),
        "audio": torch.randn(2, 3, 5),
    },
    targets=None,
)
output = model(batch)
assert output.preds.shape == (2, 2)
~~~

Ключ в реестре: tomar_fusion_model.
