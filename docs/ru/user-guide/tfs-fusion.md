# Слияние TFS

TFSFusionModel реализует вариант TripleFusion, работающий непосредственно с
признаками. Модель проецирует признаки каждой модальности в shared_dim,
вычисляет направленные межмодальные взаимодействия и выдаёт один вектор
предсказания на образец. Для трёх модальностей с группами по умолчанию
используется иерархия в стиле MASAI:

1. шесть независимых блоков перекрёстного внимания Transformer;
2. две тройки, составленные из направленных пар;
3. два взаимных блока перекрёстного внимания второго уровня;
4. маскированная агрегация по среднему и популяционному стандартному отклонению;
5. Linear -> BatchNorm -> Tanh, затем внутренний слой предсказания.

Это иерархическое слияние признаков, а не простое усреднение результатов пар.

## Параметры

~~~python
TFSFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    groups: tuple[tuple[tuple[str, str], ...], ...] | None = None,
)
~~~

- input_dims — непустое соответствие имени модальности и ширины входных
  признаков. Для каждой модальности создаётся внутренняя проекция.
- shared_dim — общая ширина блоков слияния.
- output_dim — ширина ModelOutput.preds.
- groups — необязательные группы направленных пар. По умолчанию каждая
  неупорядоченная пара представлена обоими направлениями. При заданных группах
  или другом числе модальностей используется обобщённая иерархия вместо
  трёхмодального пути MASAI.

## Входы и выходы

Batch.inputs содержит только признаки формы [B, D] или [B, T, D]. Векторы
внутри модели приводятся к одной временной позиции. ModelOutput.preds имеет
форму [B, output_dim].

Маски читаются из Batch.masks, включая legacy-вложенные маски. Выделенная
`<modality>_temporal_mask` формы [B, T] имеет приоритет; иначе
`<modality>_mask` может иметь форму [B] или [B, T]. Совместимая
sequence_mask также учитывается. Невалидные позиции не участвуют во внимании
и агрегации, а невалидные направленные пары обнуляются до слияния второго
уровня.

Направленная пара формируется только при наличии в Batch.inputs как минимум
двух тензоров заданных модальностей. Их маски при этом могут пометить все
временные позиции образца как невалидные. В таком случае pair_valid и
pair_weights равны нулю, fused заполнен нулями, а внутренний слой предсказания
всё равно возвращает preds. Если в Batch.inputs присутствует меньше двух
тензоров модальностей, модель не может сформировать пару и выбрасывает
ValueError.

В aux находятся `pair_<source>-><target>`,
`pair_valid_<source>-><target>`, pair_weights, group_weights и fused.
Основной результат находится в ModelOutput.preds.

## Python API

~~~python
from chimera_ml.models import TFSFusionModel

model = TFSFusionModel(
    input_dims={"text": 1024, "audio": 768, "video": 512},
    shared_dim=256,
    output_dim=2,
)
~~~

## Конфигурация YAML

~~~yaml
model:
  name: tfs_fusion_model
  params:
    input_dims:
      text: 1024
      audio: 768
      video: 512
    shared_dim: 256
    output_dim: 2
~~~

groups задаётся как список групп, внутри которых находятся пары из двух имён,
например [text, audio]. Пустые группы и повторяющиеся направленные пары
недопустимы. Если один tensor целиком отсутствует, может использоваться
обобщённый путь; при наличии всех трёх tensors включается иерархия по умолчанию.

## Синтетический forward

```python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import TFSFusionModel
model = TFSFusionModel({"a": 4, "b": 5, "c": 6}, shared_dim=8, output_dim=2).eval()
batch = Batch(inputs={"a": torch.randn(2, 3, 4), "b": torch.randn(2, 3, 5), "c": torch.randn(2, 3, 6)}, targets=None)
output = model(batch)
assert output.preds.shape == (2, 2)
assert "pair_weights" in output.aux
```

При отсутствии менее двух tensors модальностей возникает `ValueError`; masks и `groups` должны использовать те же names.
