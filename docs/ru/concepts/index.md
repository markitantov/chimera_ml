# Основные концепции

Модель выполнения Chimera ML намеренно компактна:

~~~text
YAML-конфигурация → функции построения реестра → BuildContext → Trainer / конвейер
                                  ↘ callbacks и loggers
~~~

- [Архитектура](architecture.md) — границы между ядром и пакетами задачи.
- [Конфигурация](configuration.md) — форма YAML эксперимента.
- [Реестры](registries.md) — построение компонентов по именам.
- [BuildContext](build-context.md) — обмен метаданными выполнения.
- [Плагины](plugins.md) — обнаружение через точки входа Python.
