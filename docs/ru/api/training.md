# API обучения

Подсистема training связывает validated experiment configuration с model,
data loaders, optimization components, metrics, loggers и callbacks. Обычно эти
компоненты задаются в YAML; direct construction полезен для integrations и
тестов. Практические сценарии описаны в [Training guide](../user-guide/training.md)
и [configuration guide](../user-guide/configuration.md).

## Configuration

\`TrainConfig\` содержит runtime settings loop: epochs, device, mixed precision,
multi-loader sampling, scheduler stepping и prediction caching.

::: chimera_ml.training.config.TrainConfig

## Trainer

\`Trainer\` управляет fit/evaluate lifecycle: переносит batches на device,
вызывает model и loss, обновляет metrics, пишет logs, вызывает callbacks и
может кэшировать split outputs для artifact callbacks.

::: chimera_ml.training.trainer.Trainer

## Build context

\`BuildContext\` передаётся между построением registry components. Components
могут читать и записывать значения и реализовывать \`describe_context\`, чтобы
обогащать context для следующих builders.

::: chimera_ml.training.builders.BuildContext

## Registry builders

\`build_from_registry\` реализует общий configuration contract \`name\` +
\`params\`. Typed helpers inject runtime dependencies — например model,
optimizer или context — и связывают YAML с Python objects.

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

\`GridSweep\` строит конечное Cartesian product или explicit trial list.
\`OptunaSweep\` передаёт typed suggestions и objective direction в Optuna.
Оба класса сохраняют base/sweep/trial configs и manifest в sweep log directory.
\`SweepTarget\` задаёт monitored metric и направление оптимизации. См. раздел
[Sweeps](../user-guide/sweeps.md).

::: chimera_ml.training.sweep.GridSweep

::: chimera_ml.training.sweep.OptunaSweep

::: chimera_ml.utils.sweep.SweepTarget
