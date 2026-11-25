# Архитектура плагинов

Плагин — обычный дистрибутив Python с вызываемой функцией регистрации в группе
точек входа `chimera_ml.plugins`. При запуске `register_all` импортирует встроенные
компоненты и один раз загружает каждую обнаруженную точку входа.

## Жизненный цикл

~~~text
pip install plugin
  → import chimera_ml.plugins.register_all()
  → discovery entry point
  → load object
  → call, если object callable
  → registry keys доступны YAML builders
~~~

Точка входа может указывать на вызываемую функцию регистрации или объект уровня
модуля. Если загруженный объект вызываем, CHIMERA-ML вызывает его; иначе достаточно
импорта ради побочных эффектов декораторов. Ошибка загрузки выдаётся как
предупреждение.

~~~bash
chimera-ml plugins list
chimera-ml registry list --type models
~~~

Практические инструкции есть в [руководстве пользователя](../user-guide/plugins.md) и
[руководстве для авторов](../development/plugins.md).
