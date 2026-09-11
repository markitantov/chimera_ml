# Training API

The training subsystem connects a validated experiment configuration to a
model, data loaders, optimization components, metrics, loggers, and callbacks.
Most applications configure these components in YAML; direct construction is
useful for library integrations and tests. See the [Training guide](../user-guide/training.md)
and [configuration guide](../user-guide/configuration.md) for workflows.

## Configuration

\`TrainConfig\` contains runtime loop settings such as epochs, device, mixed
precision, multi-loader sampling, scheduler stepping, and prediction caching.

::: chimera_ml.training.config.TrainConfig

## Trainer

\`Trainer\` owns the fit/evaluate lifecycle. It moves batches to the selected
device, calls the model and loss, updates metrics, logs values, invokes
callbacks, and can cache split outputs for artifact-producing callbacks.

::: chimera_ml.training.trainer.Trainer

## Build context

\`BuildContext\` is shared while registry components are constructed. Components
may read values, write derived values, and implement \`describe_context\` to
enrich later builders.

::: chimera_ml.training.builders.BuildContext

## Registry builders

\`build_from_registry\` implements the common \`name\` + \`params\` configuration
contract. The typed helpers inject runtime dependencies such as the model,
optimizer, or context and are the usual bridge from YAML to Python objects.

::: chimera_ml.training.builders.build_from_registry

::: chimera_ml.training.builders.build_datamodule

::: chimera_ml.training.builders.build_model

::: chimera_ml.training.builders.build_loss

::: chimera_ml.training.builders.build_metrics

::: chimera_ml.training.builders.build_optimizer

::: chimera_ml.training.builders.build_scheduler

::: chimera_ml.training.builders.build_callbacks

::: chimera_ml.training.builders.build_collate

::: chimera_ml.training.builders.build_logger

## Sweeps

\`GridSweep\` expands a finite Cartesian product or explicit trial list.
\`OptunaSweep\` delegates typed suggestions and objective direction to Optuna.
Both persist base/sweep/trial configurations and a manifest below the sweep
log directory. \`SweepTarget\` defines the monitored metric and whether lower or
higher values are preferred. See [Sweeps](../user-guide/sweeps.md).

::: chimera_ml.training.sweep.GridSweep

::: chimera_ml.training.sweep.OptunaSweep

::: chimera_ml.utils.sweep.SweepTarget
