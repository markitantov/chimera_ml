# Plugin architecture

Plugin — обычный Python distribution с registration callable в entry-point group
chimera_ml.plugins. При старте register_all импортирует built-ins и загружает
каждый обнаруженный entry point один раз.

## Lifecycle

~~~text
pip install plugin
  → import chimera_ml.plugins.register_all()
  → discovery entry point
  → load object
  → call, если object callable
  → registry keys доступны YAML builders
~~~

Entry point может указывать callable registration function или module-level
object. Если загруженный object callable, Chimera ML вызывает его; иначе
достаточно import для decorator side effects. Ошибка загрузки выдаётся как
warning.

~~~bash
chimera-ml plugins list
chimera-ml registry list --type models
~~~

Практические инструкции есть в [user guide](../user-guide/plugins.md) и
[authoring tutorial](../development/plugins.md).
