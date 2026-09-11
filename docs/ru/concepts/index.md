# Основные концепции

Модель выполнения Chimera ML намеренно компактна:

~~~text
YAML config → registry factories → BuildContext → Trainer / pipeline
                                  ↘ callbacks и loggers
~~~

- [Архитектура](architecture.md) — границы между core и task packages.
- [Конфигурация](configuration.md) — форма experiment YAML.
- [Registries](registries.md) — построение по именам.
- [BuildContext](build-context.md) — обмен runtime metadata.
- [Plugins](plugins.md) — обнаружение через Python entry points.
