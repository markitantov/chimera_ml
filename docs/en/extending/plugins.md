# Authoring a plugin

The discovery group is `chimera_ml.plugins`. A complete minimal model plugin is:

```text
my_chimera_plugin/
├── pyproject.toml
└── src/my_chimera_plugin/
    ├── __init__.py
    └── model.py
```

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "my-chimera-plugin"
version = "0.1.0"
dependencies = ["chimera-ml"]

[project.entry-points."chimera_ml.plugins"]
my_chimera_plugin = "my_chimera_plugin:register"

[tool.setuptools.packages.find]
where = ["src"]
```

`src/my_chimera_plugin/__init__.py`:

```python
from . import model

def register() -> None:
    # Importing model executes the registry decorator.
    return None
```

`src/my_chimera_plugin/model.py`:

```python
import torch
from torch import nn
from chimera_ml.core import Batch, MODELS, ModelOutput
from chimera_ml.models import BaseModel

class PluginModel(BaseModel):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.head = nn.Linear(input_dim, output_dim)
    def forward(self, batch: Batch) -> ModelOutput:
        return ModelOutput(preds=self.head(batch.inputs["features"]))

@MODELS.register("plugin_model")
def plugin_model(**params):
    return PluginModel(**params)
```

Install and verify in the CLI environment:

```bash
python -m pip install -e my_chimera_plugin
chimera-ml plugins list
chimera-ml registry list --type models
```

Then use `model: {name: plugin_model, params: {input_dim: 16, output_dim: 2}}`. The entry point is loaded in a new process; a decorator run only in an interactive Python session is not enough for `chimera-ml train`.
