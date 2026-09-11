# Inference API

Inference executes a configured sequence or dependency graph of steps. The
configuration describes the pipeline, builders resolve step factories from
\`INFERENCE_STEPS\`, each step reads/writes artifacts in an
\`InferenceContext\`, and the pipeline returns the updated context. For command
line usage, see [Inference](../user-guide/inference.md).

## Configuration

\`InferenceConfig\` exposes pipeline name, parallel mode, steps, and requested
runtime device from an inference YAML mapping.

::: chimera_ml.inference.config.InferenceConfig

## Context and graph

\`InferenceContext\` is the mutable artifact store passed to steps. An
\`InferenceGraphNode\` associates one step with its \`after\` dependencies.

::: chimera_ml.inference.context.InferenceContext

::: chimera_ml.inference.pipeline.InferenceGraphNode

## Pipeline

\`InferencePipeline\` validates node identifiers and dependencies, schedules
ready nodes, isolates step contexts, and merges written artifacts. In parallel
mode, unrelated nodes may run concurrently; conflicting artifact writes are
rejected unless the dependency graph makes the overwrite ordered.

::: chimera_ml.inference.pipeline.InferencePipeline

## Builders and utilities

Use \`build_inference_step\` for one registry-backed step and
\`build_inference_pipeline\` for a complete \`InferenceConfig\`. Device selection
accepts \`auto\`, \`cpu\`, or \`cuda\` through \`resolve_inference_device\`.

::: chimera_ml.inference.builders.build_inference_step

::: chimera_ml.inference.builders.build_inference_pipeline

::: chimera_ml.inference.utils.resolve_inference_device

## Step protocol and built-ins

Custom steps implement the small \`run(ctx) -> ctx\` protocol. The built-in
checkpoint step resolves local or HTTP(S) references into a cache; JSON steps
write or print the \`predictions\` artifact.

::: chimera_ml.inference.steps.base.BaseInferenceStep

::: chimera_ml.inference.steps.checkpoint_steps.ResolveCheckpointsStep

::: chimera_ml.inference.steps.checkpoint_steps.resolve_checkpoints_step

::: chimera_ml.inference.steps.output_steps.WriteJsonPredictionsStep

::: chimera_ml.inference.steps.output_steps.write_json_predictions_step

::: chimera_ml.inference.steps.output_steps.PrintJsonPredictionsStep

::: chimera_ml.inference.steps.output_steps.print_json_predictions_step
