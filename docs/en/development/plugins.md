# Plugin authoring tutorial

This is the contributor-oriented version of [Writing a plugin](../user-guide/plugins.md).

## Minimal package

The repository examples use a src layout and a top-level chimera_plugin.py:

~~~text
my-plugin/
├── pyproject.toml
└── src/
    ├── chimera_plugin.py
    └── components.py
~~~

~~~toml
[project.entry-points."chimera_ml.plugins"]
my_plugin = "chimera_plugin:register"
~~~

~~~python
def register():
    import components
~~~

In components.py, import the registry and decorate keyword-oriented factories:

~~~python
from chimera_ml.core import MODELS


@MODELS.register("my_model")
def my_model(*, hidden_dim=128, context=None):
    return MyModel(hidden_dim=hidden_dim)
~~~

Use context only when the declared factory needs runtime metadata. For an
inference component, register a factory in INFERENCE_STEPS that returns an
object with run(ctx) and writes outputs with ctx.set_artifact.

## Test and install

Install editable, inspect the entry point, then inspect the expected registry:

~~~bash
python -m pip install -e my-plugin
chimera-ml plugins list
chimera-ml registry list --type models
~~~

Write tests for registration, factory parameters, and the component's runtime
boundary. The VA, ORAGEN, and affective states packages are reference
implementations in examples/.
