# Plugin architecture

Plugins are ordinary Python distributions. They expose a registration callable
through the `chimera_ml.plugins` entry-point group. On startup, `register_all`
imports built-in modules and then loads each discovered entry point once.

## Lifecycle

```text
pip install plugin
  → import chimera_ml.plugins.register_all()
  → discover entry point
  → load object
  → call it when callable
  → registry keys become available to YAML builders
```

An entry point may target a callable registration function or a module-level
object. If the loaded object is callable, Chimera ML calls it; otherwise
loading the module is enough for decorator side effects. Plugin loading
failures emit a warning and do not replace the core process with a traceback.

Inspect discovery and registration separately:

```bash
chimera-ml plugins list
chimera-ml registry list --type models
```

The [plugin how-to](../user-guide/plugins.md) and [authoring tutorial](../development/plugins.md)
show a real package layout based on the examples in this repository.
