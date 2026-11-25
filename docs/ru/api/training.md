# API обучения

Подсистема обучения связывает проверенную конфигурацию эксперимента с моделью,
загрузчиками данных, компонентами оптимизации, метриками, логгерами и колбэками.
Обычно эти компоненты задаются в YAML; прямое создание полезно для интеграций и
тестов. Практические сценарии описаны в [руководстве по обучению](../user-guide/training.md)
и [руководстве по конфигурации](../user-guide/configuration.md).

## Конфигурация

`TrainConfig` содержит параметры цикла выполнения: эпохи, устройство, смешанную точность,
выборку из нескольких загрузчиков, шаги планировщика и кэширование предсказаний.

::: chimera_ml.training.config.TrainConfig

## Trainer

`Trainer` управляет жизненным циклом обучения и оценки: переносит батчи на устройство,
вызывает модель и функцию потерь, обновляет метрики, записывает журналы, вызывает колбэки
и может кэшировать результаты выборок для колбэков артефактов.

::: chimera_ml.training.trainer.Trainer

## Контекст построения

\`BuildContext\` передаётся между этапами построения компонентов реестра. Компоненты
могут читать и записывать значения и реализовывать \`describe_context\`, чтобы
обогащать контекст для следующих функций построения.

::: chimera_ml.training.builders.BuildContext

## Функции построения реестров

\`build_from_registry\` реализует общий контракт конфигурации \`name\` +
\`params\`. Типизированные вспомогательные функции внедряют зависимости выполнения — например модель,
оптимизатор или контекст — и связывают YAML с объектами Python.

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

## Переборы

\`GridSweep\` строит конечное декартово произведение или явный список испытаний.
\`OptunaSweep\` передаёт типизированные предложения и направление целевой функции в Optuna.
Оба класса сохраняют базовую конфигурацию, конфигурацию перебора и конфигурации испытаний, а также манифест в каталоге журналов перебора.
\`SweepTarget\` задаёт контролируемую метрику и направление оптимизации. См. раздел
[Переборы](../user-guide/sweeps.md).

::: chimera_ml.training.sweep.GridSweep

::: chimera_ml.training.sweep.OptunaSweep

::: chimera_ml.utils.sweep.SweepTarget
