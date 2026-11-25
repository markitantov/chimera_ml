# Пользовательские шаги предсказания

Шаг реализует протокол `run(ctx: InferenceContext) -> InferenceContext`; наследование от базового класса не требуется. Он читает `ctx.input_path`, `ctx.device` и upstream-артефакты, а результаты публикует через `ctx.set_artifact`.

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

Для CLI положите factory в устанавливаемый плагин и объявите entry point:

```toml
[project.entry-points."chimera_ml.plugins"]
my_inference_plugin = "my_inference_plugin:register"
```

```python
from . import steps

def register():
    return None
```

После `python -m pip install -e path/to/plugin` проверьте `chimera-ml plugins list` и `chimera-ml registry list --type inference_steps`. Для XAI prediction step должен предоставлять вызываемый `on_prediction`; см. [раздел XAI](../user-guide/xai.md).
