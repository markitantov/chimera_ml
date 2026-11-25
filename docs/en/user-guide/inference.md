# Inference overview

Inference is an independent, configuration-driven execution mechanism. It runs an `InferencePipeline` over one input path and a shared `InferenceContext`; it is not a training or evaluation shortcut.

```text
Input
  -> InferenceContext
  -> input preparation
  -> checkpoint resolution
  -> plugin prediction step
  -> optional XAI
  -> postprocessing
  -> output artifacts
```

This is a conceptual flow, not a mandatory fixed pipeline. A real application chooses the steps it needs.

## Training, evaluation, and inference

| Operation | Runtime | Updates parameters? | Typical input/output |
| --- | --- | --- | --- |
| Training | `Trainer.fit` | Yes | DataLoaders with targets -> checkpoints and logs |
| Evaluation | `Trainer.evaluate` via `chimera-ml eval` | No | Configured splits + compatible checkpoint -> metrics/cache |
| Inference | `InferencePipeline` via `chimera-ml inference` | No | One input path -> shared artifacts and optional JSON |

Training YAML sections such as `data`, `loss`, `optimizer`, and `train` do not configure an inference pipeline. Inference YAML has `pipeline`, optional `runtime`, and `steps` instead. `Trainer.evaluate` is for labeled split metrics; inference is for application execution and prediction export.

## What the framework and application provide

The core provides `InferenceConfig`, `InferenceContext`, pipeline scheduling, the `INFERENCE_STEPS` registry, checkpoint resolution, JSON output, and the universal XAI wrapper. The application or plugin provides input preparation, feature extraction, model construction, checkpoint loading, and prediction steps for a particular task. There is no universal core prediction step for every model.

Steps communicate through the context:

- `ctx.input_path` is the CLI input path;
- `ctx.work_dir` is the persistent or temporary working directory;
- `ctx.device` is the resolved `cpu` or `cuda` string;
- `ctx.get_artifact(name, default)` reads upstream values;
- `ctx.set_artifact(name, value)` publishes a value and records ownership for DAG merging;
- `ctx.predictions` is the conventional `predictions` artifact.

A prediction step normally loads a model in evaluation mode, constructs `Batch` objects from prepared data, runs `ModelOutput` under `torch.inference_mode()`, and publishes JSON-compatible predictions. Output steps then print or write those predictions.

Continue with [Run inference](../inference/run.md), [pipeline configuration](../inference/pipeline.md), [built-in steps](../inference/built-in-steps.md), and [XAI](xai.md).
