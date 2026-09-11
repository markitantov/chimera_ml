# Writing a plugin

A plugin is a Python package exposing a registration function through the
chimera_ml.plugins entry-point group.

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
    import my_components  # imports the registry decorators
~~~

Register models, datamodules, losses, metrics, callbacks, collates, loggers,
or inference steps with the corresponding registry. Accept context only when
the factory needs BuildContext metadata.

Install and inspect:

~~~bash
python -m pip install -e my-plugin
chimera-ml plugins list
chimera-ml registry list --type models
~~~

Test registration and behavior in the plugin project. Duplicate keys are
rejected; plugin load failures are warnings, so check expected keys explicitly.
