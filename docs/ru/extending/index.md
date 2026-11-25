# Расширение CHIMERA-ML

Автор extension реализует task-specific component за публичным contract и регистрирует factory под стабильным key. Dataset, model, loss и callbacks должны жить в Python package/plugin, а не требовать правок core source.

- [Registries](registries.md) — lookup, factory и duplicate keys.
- [BuildContext](build-context.md) — construction metadata, не inference artifacts.
- [Plugins](plugins.md) — полный installable plugin.
- [Custom data](data.md), [models](models.md), [training components](training.md), [callbacks](callbacks.md).
- Для inference используйте полный [custom-step guide](../inference/custom-steps.md) и packaging из [Plugins](plugins.md).

Основные contracts малы: dataset item — mapping, model — `Batch → ModelOutput`, loss — callable output/batch, metric — `reset/update/compute`, callback — lifecycle hooks.
