# API core

Core types — общий vocabulary для data modules, models, losses, metrics и
builders. \`Batch\` содержит modality tensors, optional targets, masks и metadata;
\`ModelOutput\` содержит predictions и auxiliary tensors. \`ExperimentConfig\`
оборачивает YAML и поддерживает validation и dotted overrides. Практические
примеры находятся в [configuration guide](../user-guide/configuration.md).

## Batch и model output

::: chimera_ml.core.batch.Batch

::: chimera_ml.core.types.ModelOutput

## Experiment configuration

::: chimera_ml.core.config.ExperimentConfig

::: chimera_ml.core.config.load_yaml

## Registries

\`Registry\` связывает stable configuration key с factory или class. Все
training, data, logging и inference registries используют этот API; обычно
components создаются соответствующим builder.

::: chimera_ml.core.registry.Registry
