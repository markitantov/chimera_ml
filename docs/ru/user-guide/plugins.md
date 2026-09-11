# Writing a plugin

Plugin — Python package с registration function в entry-point group
chimera_ml.plugins.

~~~text
my-plugin/
├── pyproject.toml
└── src/
    ├── chimera_plugin.py
    └── my_components.py
~~~

~~~toml
[project.entry-points."chimera_ml.plugins"]
my_plugin = "chimera_plugin:register"
~~~

~~~python
from chimera_ml.core import MODELS


def register():
    import my_components
~~~

Регистрируйте models, datamodules, losses, metrics, callbacks, collates,
loggers или inference steps в соответствующем registry. Принимайте context
только если factory использует BuildContext.

~~~bash
python -m pip install -e my-plugin
chimera-ml plugins list
chimera-ml registry list --type models
~~~

Пишите tests для discovery, factory params и runtime boundary. Duplicate keys
отклоняются; plugin load failures являются warnings.
