# Создание plugin

Discovery group — `chimera_ml.plugins`. Минимальный model plugin:

```text
my_chimera_plugin/
├── pyproject.toml
└── src/my_chimera_plugin/
    ├── __init__.py
    └── model.py
```

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

```python
# src/my_chimera_plugin/__init__.py
from . import model
def register() -> None:
    return None
```

```python
# src/my_chimera_plugin/model.py
from torch import nn
from chimera_ml.core import Batch, MODELS, ModelOutput
from chimera_ml.models import BaseModel
class PluginModel(BaseModel):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__(); self.head = nn.Linear(input_dim, output_dim)
    def forward(self, batch: Batch) -> ModelOutput:
        return ModelOutput(preds=self.head(batch.inputs["features"]))
@MODELS.register("plugin_model")
def plugin_model(**params): return PluginModel(**params)
```

```bash
python -m pip install -e my_chimera_plugin
chimera-ml plugins list
chimera-ml registry list --type models
```

После установки YAML может использовать `model: {name: plugin_model, params: {input_dim: 16, output_dim: 2}}`. Entry point загружается в новом CLI process; decorator в interactive session недостаточен.
