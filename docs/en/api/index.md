# API reference

The API reference explains the contracts used by the framework subsystems. It
is generated from the installed package with mkdocstrings, while each page
adds curated context, configuration-to-object relationships, and links to
task-oriented guidance.

Start with [Core](core.md) for Batch, ModelOutput, ExperimentConfig, and
Registry. Then choose the subsystem that matches the component being built or
configured.

## Subsystems

- [Core](core.md): shared containers, configuration, and registry primitives.
- [Data](data.md): DataModule, collation, masks, and loader normalization.
- [Models](models.md): model contract and multimodal fusion implementations.
- [Training](training.md): Trainer, builders, BuildContext, and sweeps.
- [Inference](inference.md): contexts, DAG pipelines, steps, and builders.
- [Callbacks](callbacks.md): lifecycle extensions and artifact callbacks.
- [Losses](losses.md): optimization objectives and registry factories.
- [Metrics](metrics.md): stateful epoch metrics and aggregation.
- [Logging](logging.md): console/file and MLflow logger contracts.
- [Plugins](plugins.md): built-in and entry-point registration.

Use the [User Guide](../user-guide/index.md) when you need to accomplish a
task; use these pages when you need exact signatures, accepted fields, return
values, and extension contracts.
