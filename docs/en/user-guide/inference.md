# Inference

Inference is a separate, configurable execution pipeline. It does not call `Trainer.evaluate`: it creates an `InferenceContext`, runs registered steps, and passes named artifacts between them.

~~~bash
chimera-ml inference -i input.mp4 -o prediction.json -c inference.yaml --device auto --work-dir work
~~~

`--input` and `--config-path` are required. `--device` accepts `auto`, `cpu`, or `cuda`; `auto` selects CUDA when available. If `--work-dir` is omitted, the CLI creates a temporary working directory.

## Configuration

~~~yaml
pipeline:
  name: demo
  parallel: false
runtime:
  device: auto
steps:
  - name: resolve_checkpoints_step
    params:
      cache_dir: checkpoints
      checkpoints:
        model: /absolute/path/to/model.pt
  - name: my_prediction_step
    params:
      checkpoint_key: model
  - name: write_json_predictions_step
    params:
      output_path: prediction.json
~~~

Each step has a registry `name`, optional `id`, optional `after` dependencies, and optional `params`. A step implements `run(ctx: InferenceContext) -> InferenceContext`; it reads `ctx.input_path`, `ctx.device`, and artifacts and publishes results with `ctx.set_artifact`. The `predictions` property is a convenience view of the `predictions` artifact.

## Built-in steps

The core package registers these public steps:

| Key | Parameters | Inputs and outputs |
| --- | --- | --- |
| `resolve_checkpoints_step` | `checkpoints`, `cache_dir="checkpoints"`, `force_download=false`, `chunk_size=8192` | Accepts local or HTTP(S) checkpoint references; writes absolute paths to `artifacts["checkpoints"]` and the cache path to `artifacts["cache_dir"]`. |
| `write_json_predictions_step` | `output_path=null` | Requires `predictions`; writes `input` and `predictions` JSON and publishes `output_path`. |
| `print_json_predictions_step` | none | Requires `predictions`; prints the same JSON payload and writes no file. |
| `xai_step` | nested `step`, `modalities`, `ablation_policy`, and optional score/diagnostic settings | Wraps a compatible prediction step and publishes `xai`; see [XAI](xai.md). |

Prediction, feature-extraction, and model-specific steps in the examples are plugin components, not built-ins. List the actual registrations in the current environment with `chimera-ml registry list --type inference_steps`.

## Sequential and parallel pipelines

With `parallel: false` or no `parallel` field, steps run in listed order. The CLI `--output` can create or override `write_json_predictions_step` only in this mode. With `parallel: true`, every step is a DAG node and `after` is authoritative. Steps without `after` are roots and may start together; repeated names need explicit `id` values.

Unknown dependencies, duplicate IDs, self-dependencies, cycles, and conflicting writes to the same explicitly-set artifact are errors. A branch receives a copy of upstream artifacts; downstream steps merge only keys written through `set_artifact`. Declare the output step explicitly in parallel mode and include its dependencies.

## Checkpoints and artifacts

`resolve_checkpoints_step` reuses cached downloads unless `force_download: true`; incomplete downloads are removed. Downstream plugin steps must agree on the checkpoint key in `artifacts["checkpoints"]`. Inference output is owned by explicit output steps, so a successful pipeline without a writer may produce artifacts without a JSON file.

For a custom prediction step, follow the minimal callback-compatible example in [XAI](xai.md) and register it through a plugin entry point.
