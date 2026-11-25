# Пользовательские шаги предсказания

Шаг реализует протокол `run(ctx: InferenceContext) -> InferenceContext`; наследование от базового класса не требуется. Он читает `ctx.input_path`, `ctx.device` и артефакты предыдущих шагов, а результаты публикует через `ctx.set_artifact`.

```python
from dataclasses import dataclass
import torch
from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.inference import InferenceContext

@dataclass
class ToyPredictionStep:
    value: float = 3.0

    def run(self, ctx: InferenceContext) -> InferenceContext:
        x = torch.tensor([[self.value]], device=ctx.device)
        batch = Batch(inputs={"x": x}, targets=None, meta={"sample_ids": ["toy-0"]})
        with torch.inference_mode():
            output = ModelOutput(preds=x * 2.0)
        ctx.set_artifact("predictions", output.preds.cpu().tolist())
        return ctx

@INFERENCE_STEPS.register("toy_prediction_step")
def toy_prediction_step(**params):
    return ToyPredictionStep(**params)
```

## Полный детерминированный конвейер через Python API

Этот пример не требует набора данных, контрольной точки, учётных данных или внешнего плагина: он регистрирует синтетический шаг в текущем процессе, строит конвейер и проверяет средство записи JSON.

```python
from dataclasses import dataclass
from pathlib import Path
import tempfile

import torch
from torch import nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.inference import InferenceConfig, InferenceContext, build_inference_pipeline

class ToyModel(nn.Module):
    def forward(self, batch: Batch) -> ModelOutput:
        return ModelOutput(preds=batch.inputs["x"] * 2.0)

@dataclass
class ToyPredictionStep:
    value: float = 3.0

    def run(self, ctx: InferenceContext) -> InferenceContext:
        model = ToyModel().to(ctx.device).eval()
        batch = Batch(
            inputs={"x": torch.tensor([[self.value]], device=ctx.device)},
            targets=None,
            meta={"sample_ids": ["toy-0"]},
        )
        with torch.inference_mode():
            output = model(batch)
        ctx.set_artifact("predictions", output.preds.detach().cpu().tolist())
        return ctx

@INFERENCE_STEPS.register("toy_prediction_step")
def toy_prediction_step(**params):
    return ToyPredictionStep(**params)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    output = root / "prediction.json"
    config = InferenceConfig({
        "pipeline": {"name": "toy-demo"},
        "steps": [
            {"name": "toy_prediction_step", "params": {"value": 3.0}},
            {"name": "write_json_predictions_step", "params": {"output_path": str(output)}},
        ],
    })
    context = InferenceContext(root / "input.bin", root, "cpu", config.raw)
    build_inference_pipeline(config).run(context)
    assert context.predictions == [[6.0]]
    assert output.exists()
```

Здесь `Batch`, `ModelOutput`, `model.eval()`, `torch.inference_mode()` и `set_artifact` соответствуют публичным контрактам. Средство записи создаёт JSON только с `input` и `predictions`; произвольные артефакты контекста в него не попадают.

Для CLI поместите фабрику в устанавливаемый плагин и объявите точку входа:

```toml
[project.entry-points."chimera_ml.plugins"]
my_inference_plugin = "my_inference_plugin:register"
```

```python
from . import steps

def register():
    return None
```

Декоратор действует только в текущем процессе Python. Для обнаружения через CLI упакуйте шаг в плагин, поместите его фабрику в модуль `steps.py` и объявите точку входа.

## Доступность шага в CLI

После `python -m pip install -e path/to/plugin` проверьте `chimera-ml plugins list` и `chimera-ml registry list --type inference_steps`. Для XAI шаг предсказания должен предоставлять вызываемый `on_prediction`; см. [раздел XAI](../user-guide/xai.md).
