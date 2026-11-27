# API инференса

Инференс выполняет настроенную последовательность шагов или граф зависимостей. Конфигурация
описывает конвейер, функции построения получают фабрики шагов из `INFERENCE_STEPS`,
каждый шаг читает и записывает артефакты в `InferenceContext`, а конвейер
возвращает обновлённый контекст. Сценарии CLI описаны в
[разделе об инференсе](../user-guide/inference.md).

## Конфигурация

`InferenceConfig` предоставляет имя конвейера, параллельный режим, шаги и
запрошенное устройство выполнения из словаря YAML-конфигурации инференса.

::: chimera_ml.inference.config.InferenceConfig

## Контекст и граф

`InferenceContext` — изменяемое хранилище артефактов, передаваемое шагам.
`InferenceGraphNode` связывает один шаг с его зависимостями из `after`.

::: chimera_ml.inference.context.InferenceContext

::: chimera_ml.inference.pipeline.InferenceGraphNode

## Конвейер

`InferencePipeline` проверяет идентификаторы узлов и зависимости, планирует готовые
узлы, изолирует контексты шагов и объединяет записанные артефакты. В параллельном
режиме независимые узлы могут выполняться одновременно; конфликтующие записи
отклоняются, если граф зависимостей не задаёт порядок перезаписи.

::: chimera_ml.inference.pipeline.InferencePipeline

## Функции построения и вспомогательные функции

Используйте `build_inference_step` для одного шага, построенного через реестр, и
`build_inference_pipeline` для полного `InferenceConfig`. Выбор устройства
(`auto`, `cpu`, `cuda`) выполняет `resolve_inference_device`.

::: chimera_ml.inference.builders.build_inference_step

::: chimera_ml.inference.builders.build_inference_pipeline

::: chimera_ml.inference.utils.resolve_inference_device

## Объяснимый ИИ (XAI)

Компоненты XAI измеряют чувствительность зарегистрированного шага предсказания
к исключению одной модальности. `XAIStep` оборачивает шаг, который вызывает
`on_prediction` после каждого прямого прохода модели, `explain_batch` выполняет
исходный прогон и вмешательства, `summarize_explanations` усредняет знаковые и
абсолютные изменения, а `xai_step` является фабрикой из реестра. Подробности
приведены в [руководстве по XAI](../user-guide/xai.md).

::: chimera_ml.inference.steps.xai_steps.XAIStep

::: chimera_ml.inference.steps.xai_steps.explain_batch

::: chimera_ml.inference.steps.xai_steps.summarize_explanations

::: chimera_ml.inference.steps.xai_steps.xai_step

## Протокол шагов и встроенные компоненты

Пользовательский шаг реализует небольшой протокол `run(ctx) -> ctx`. Встроенный
шаг контрольных точек разрешает локальные ссылки или ссылки HTTP(S) в кэше; шаги
JSON записывают или печатают артефакт `predictions`.

::: chimera_ml.inference.steps.base.BaseInferenceStep

::: chimera_ml.inference.steps.checkpoint_steps.ResolveCheckpointsStep

::: chimera_ml.inference.steps.checkpoint_steps.resolve_checkpoints_step

::: chimera_ml.inference.steps.output_steps.WriteJsonPredictionsStep

::: chimera_ml.inference.steps.output_steps.write_json_predictions_step

::: chimera_ml.inference.steps.output_steps.PrintJsonPredictionsStep

::: chimera_ml.inference.steps.output_steps.print_json_predictions_step
