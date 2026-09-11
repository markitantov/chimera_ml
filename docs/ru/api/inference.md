# API inference

Inference выполняет configured sequence или dependency graph steps. Configuration
описывает pipeline, builders получают step factories из \`INFERENCE_STEPS\`,
каждый step читает и записывает artifacts в \`InferenceContext\`, а pipeline
возвращает обновлённый context. CLI-сценарии описаны в
[Inference](../user-guide/inference.md).

## Configuration

\`InferenceConfig\` предоставляет pipeline name, parallel mode, steps и
requested runtime device из inference YAML mapping.

::: chimera_ml.inference.config.InferenceConfig

## Context и graph

\`InferenceContext\` — mutable artifact store, передаваемый steps.
\`InferenceGraphNode\` связывает один step с его dependencies из \`after\`.

::: chimera_ml.inference.context.InferenceContext

::: chimera_ml.inference.pipeline.InferenceGraphNode

## Pipeline

\`InferencePipeline\` проверяет node identifiers и dependencies, планирует ready
nodes, изолирует step contexts и объединяет записанные artifacts. В parallel
mode независимые nodes могут выполняться concurrently; conflicting writes
отклоняются, если dependency graph не задаёт порядок overwrite.

::: chimera_ml.inference.pipeline.InferencePipeline

## Builders и utilities

Используйте \`build_inference_step\` для одного registry-backed step и
\`build_inference_pipeline\` для полного \`InferenceConfig\`. Выбор device
(\`auto\`, \`cpu\`, \`cuda\`) выполняет \`resolve_inference_device\`.

::: chimera_ml.inference.builders.build_inference_step

::: chimera_ml.inference.builders.build_inference_pipeline

::: chimera_ml.inference.utils.resolve_inference_device

## Step protocol и built-ins

Custom step реализует небольшой protocol \`run(ctx) -> ctx\`. Встроенный
checkpoint step разрешает local или HTTP(S) references в cache; JSON steps
записывают или печатают artifact \`predictions\`.

::: chimera_ml.inference.steps.base.BaseInferenceStep

::: chimera_ml.inference.steps.checkpoint_steps.ResolveCheckpointsStep

::: chimera_ml.inference.steps.checkpoint_steps.resolve_checkpoints_step

::: chimera_ml.inference.steps.output_steps.WriteJsonPredictionsStep

::: chimera_ml.inference.steps.output_steps.write_json_predictions_step

::: chimera_ml.inference.steps.output_steps.PrintJsonPredictionsStep

::: chimera_ml.inference.steps.output_steps.print_json_predictions_step
