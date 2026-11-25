# API инференса

Инференс выполняет настроенную последовательность шагов или граф зависимостей. Конфигурация
описывает pipeline, builders получают step factories из \`INFERENCE_STEPS\`,
каждый шаг читает и записывает артефакты в \`InferenceContext\`, а pipeline
возвращает обновлённый context. CLI-сценарии описаны в
[Inference](../user-guide/inference.md).

## Конфигурация

\`InferenceConfig\` предоставляет pipeline name, parallel mode, steps и
requested runtime device из inference YAML mapping.

::: chimera_ml.inference.config.InferenceConfig

## Контекст и граф

\`InferenceContext\` — mutable artifact store, передаваемый steps.
\`InferenceGraphNode\` связывает один step с его dependencies из \`after\`.

::: chimera_ml.inference.context.InferenceContext

::: chimera_ml.inference.pipeline.InferenceGraphNode

## Конвейер

\`InferencePipeline\` проверяет node identifiers и dependencies, планирует ready
nodes, изолирует step contexts и объединяет записанные artifacts. В parallel
mode независимые nodes могут выполняться concurrently; conflicting writes
отклоняются, если dependency graph не задаёт порядок overwrite.

::: chimera_ml.inference.pipeline.InferencePipeline

## Строители и утилиты

Используйте \`build_inference_step\` для одного registry-backed step и
\`build_inference_pipeline\` для полного \`InferenceConfig\`. Выбор device
(\`auto\`, \`cpu\`, \`cuda\`) выполняет \`resolve_inference_device\`.

::: chimera_ml.inference.builders.build_inference_step

::: chimera_ml.inference.builders.build_inference_pipeline

::: chimera_ml.inference.utils.resolve_inference_device

## Протокол шагов и встроенные компоненты

Пользовательский шаг реализует небольшой протокол \`run(ctx) -> ctx\`. Встроенный
checkpoint- шаг разрешает локальные ссылки или ссылки HTTP(S) в кэше; Шаги JSON
записывают или печатают artifact \`predictions\`.

::: chimera_ml.inference.steps.base.BaseInferenceStep

::: chimera_ml.inference.steps.checkpoint_steps.ResolveCheckpointsStep

::: chimera_ml.inference.steps.checkpoint_steps.resolve_checkpoints_step

::: chimera_ml.inference.steps.output_steps.WriteJsonPredictionsStep

::: chimera_ml.inference.steps.output_steps.write_json_predictions_step

::: chimera_ml.inference.steps.output_steps.PrintJsonPredictionsStep

::: chimera_ml.inference.steps.output_steps.print_json_predictions_step
