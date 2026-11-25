# Адаптивное слияние TFS-DCMMOE

`TFSDCMMOEFusionModel` сохраняет иерархию TFS и добавляет зависящие от входа шлюзы над экспертами направленных пар. Для каждой доступной пары вычисляется оценка, а softmax с маской исключает недоступные пары до применения весов. Поэтому модель может выбирать межмодальные связи отдельно для каждого образца.

Для трёх модальностей и групп по умолчанию используются те же шесть направленных блоков Transformer первого уровня и два взаимных блока второго уровня, что и в TFS. Обычный TFS использует бинарную доступность пары, а DCMMOE применяет обучаемые нормированные веса, сохраняя нулевой вес для недоступных экспертов.

## Параметры

~~~python
TFSDCMMOEFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    groups: tuple[tuple[tuple[str, str], ...], ...] | None = None,
    gate_hidden: int = 64,
)
~~~

`input_dims`, `shared_dim`, `output_dim` и `groups` имеют тот же смысл, что в TFS. `gate_hidden` — ширина скрытого слоя MLP, вычисляющего веса пар. Размерности должны быть положительными.

## Входы, маски и выходы

В `Batch.inputs` передаются только признаки формы `[B, D]` или `[B, T, D]`; модель сама проецирует их в `shared_dim`. Предсказание имеет форму `[B, output_dim]`. Явные временные маски имеют приоритет над устаревшими масками наличия, а полностью замаскированные пары получают нулевой вес.

Направленная пара формируется только при наличии в `Batch.inputs` как минимум двух тензоров заданных модальностей. Их маски при этом могут пометить все временные позиции образца как невалидные. В таком случае `pair_valid` и `pair_weights` равны нулю, `fused` заполнен нулями, а внутренний слой предсказания всё равно возвращает `preds`. Если в `Batch.inputs` присутствует меньше двух тензоров модальностей, модель не может сформировать пару и выбрасывает `ValueError`.

В `ModelOutput.aux` находятся представления пар и признаки их валидности, а также `pair_weights`, `group_weights` и `fused`. `pair_weights` показывает адаптивные веса экспертов с учётом масок; результат задачи находится в `preds`.

## Python API

~~~python
from chimera_ml.models import TFSDCMMOEFusionModel

model = TFSDCMMOEFusionModel(
    input_dims={"text": 1024, "audio": 768, "video": 512},
    shared_dim=256,
    output_dim=2,
    gate_hidden=128,
)
~~~

## Конфигурация YAML

~~~yaml
model:
  name: tfs_dcmmmoe_fusion_model
  params:
    input_dims:
      text: 1024
      audio: 768
      video: 512
    shared_dim: 256
    output_dim: 2
    gate_hidden: 128
~~~

`groups` позволяет задать явную иерархию групп. При нестандартных `groups` или менее чем трёх именах используется обобщённый путь TFS, но адаптивное взвешивание пар с учётом масок сохраняется.

## Синтетический forward

```python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import TFSDCMMOEFusionModel
model = TFSDCMMOEFusionModel({"a": 4, "b": 5, "c": 6}, shared_dim=8, output_dim=2, gate_hidden=16).eval()
batch = Batch(inputs={"a": torch.randn(2, 3, 4), "b": torch.randn(2, 3, 5), "c": torch.randn(2, 3, 6)}, targets=None)
output = model(batch)
assert output.preds.shape == (2, 2)
assert output.aux["pair_weights"].shape[0] == 2
```

Шлюз выбирает `pair_weights` отдельно для каждого образца, но не восстанавливает отсутствующий тензор; менее двух доступных модальностей приводят к `ValueError`.
