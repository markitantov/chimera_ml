# Custom prediction steps

A prediction step implements the small public protocol `run(ctx: InferenceContext) -> InferenceContext`. It does not need to subclass a framework class. It may read `ctx.input_path`, `ctx.device`, and upstream artifacts; it must return a context and should publish JSON-compatible results through `ctx.set_artifact`.

## Complete deterministic Python API example

This example uses no dataset, checkpoint, credentials, or external plugin. It registers a deterministic PyTorch prediction step in the current Python process, builds a pipeline, and writes JSON:

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
    config = InferenceConfig(
        {
            "pipeline": {"name": "toy-demo"},
            "steps": [
                {"name": "toy_prediction_step", "params": {"value": 3.0}},
                {"name": "write_json_predictions_step", "params": {"output_path": str(output)}},
            ],
        }
    )
    context = InferenceContext(
        input_path=root / "input.bin",
        work_dir=root,
        device="cpu",
        config=config.raw,
    )
    build_inference_pipeline(config).run(context)
    assert context.predictions == [[6.0]]
    assert output.exists()
```

The example uses `Batch`, `ModelOutput`, `model.eval()`, `torch.inference_mode()`, and `set_artifact`. The writer then creates the `{input, predictions}` JSON payload.

## Making the step available to the CLI

The decorator only affects the current Python process. For CLI discovery, package the step and expose an entry point:

```text
my_inference_plugin/
├── pyproject.toml
└── src/
    └── my_inference_plugin/
        ├── __init__.py
        └── steps.py
```

```toml
[project.entry-points."chimera_ml.plugins"]
my_inference_plugin = "my_inference_plugin:register"
```

```python
# my_inference_plugin/__init__.py
from . import steps

def register():
    return None
```

Put the `ToyPredictionStep` and its registered factory in `steps.py`, install with `python -m pip install -e my_inference_plugin`, and verify:

```bash
chimera-ml plugins list
chimera-ml registry list --type inference_steps
```

Then the YAML can refer to `toy_prediction_step`. A prediction step that does not need XAI can stop here. For XAI compatibility, add an optional `on_prediction` callable and invoke it after the model forward; see [XAI](../user-guide/xai.md).
