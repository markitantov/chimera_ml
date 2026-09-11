# Plugin authoring tutorial

Это contributor-версия [Writing a plugin](../user-guide/plugins.md).

## Minimal package

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

Factory регистрируется decorator-ом:

~~~python
from chimera_ml.core import MODELS


@MODELS.register("my_model")
def my_model(*, hidden_dim=128, context=None):
    return MyModel(hidden_dim=hidden_dim)
~~~

Inference component регистрируется в INFERENCE_STEPS, возвращает object с
run(ctx) и пишет outputs через ctx.set_artifact. Для context используйте
только declared runtime metadata.

## Test и install

~~~bash
python -m pip install -e my-plugin
chimera-ml plugins list
chimera-ml registry list --type models
~~~

Пишите tests для registration, factory parameters и runtime boundary. Reference
implementations находятся в examples/.
