# Extending CHIMERA-ML

Extension authors implement task-specific components behind the public contracts and register factories under stable keys. Keep application code in a Python package; do not modify core source just to add a dataset or model.

- [Registries](registries.md) explains lookup and duplicate-key behavior.
- [BuildContext](build-context.md) explains dependency metadata during construction; it is distinct from `InferenceContext`.
- [Plugins](plugins.md) shows a complete installable plugin.
- [Custom data](data.md), [custom models](models.md), [training components](training.md), and [callbacks](callbacks.md) give implementation templates.
- For inference steps, use the complete [Inference custom-step guide](../inference/custom-steps.md), then add the plugin packaging details from [Plugins](plugins.md).

Extension contracts are intentionally small: dataset items are mappings, models accept `Batch` and return `ModelOutput`, losses are callable on output/batch, metrics implement `reset/update/compute`, and callbacks implement lifecycle hooks. Register only public factories and validate task-specific shapes in the component.
