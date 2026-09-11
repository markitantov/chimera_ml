# Core API

Core types are the stable vocabulary shared by data modules, models, losses,
metrics, and builders. \`Batch\` carries modality tensors and optional targets,
masks, and metadata; \`ModelOutput\` carries predictions and auxiliary tensors.
\`ExperimentConfig\` wraps YAML and supports validation and dotted overrides.
The [configuration guide](../user-guide/configuration.md) explains how these
objects appear in an experiment.

## Batch and model output

::: chimera_ml.core.batch.Batch

::: chimera_ml.core.types.ModelOutput

## Experiment configuration

::: chimera_ml.core.config.ExperimentConfig

::: chimera_ml.core.config.load_yaml

## Registries

\`Registry\` maps a stable configuration key to a factory or class. Training,
data, logging, and inference registries all use the same API; components are
usually created through the relevant builder.

::: chimera_ml.core.registry.Registry
