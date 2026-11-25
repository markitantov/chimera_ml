# Pipeline configuration

Inference YAML is separate from training YAML. `InferenceConfig` reads three areas:

| Field | Type/default | Meaning |
| --- | --- | --- |
| `pipeline.name` | string, default `inference_pipeline` | Human-readable pipeline name printed by the CLI. |
| `pipeline.parallel` | boolean, default `false` | Enables explicit DAG scheduling. Invalid non-boolean values raise `TypeError`. |
| `runtime.device` | string, default `auto` | Requested runtime device; CLI resolves it to `cpu` or `cuda`. |
| `steps` | list, default `[]` | Ordered step mappings. Each step needs a non-empty `name`; `params` is optional. |
| `steps[].id` | string, default `name` | Unique node identifier used by `after`; needed when the same step name is reused. |
| `steps[].after` | string/list of strings, default `[]` | Explicit dependencies in parallel mode. |
| `steps[].params` | mapping, default `{}` | Keyword arguments passed to the registered factory. |

Minimal shape:

```yaml
pipeline:
  name: demo
  parallel: false
runtime:
  device: auto
steps:
  - name: print_json_predictions_step
    params: {}
```

The print step still requires an upstream `predictions` artifact, so the minimal shape is structurally valid but not executable by itself.

## CLI overrides and output

The CLI flags do not rewrite arbitrary YAML fields. `--device` overrides the configured runtime device for that invocation. In sequential mode, `--output` creates a writer if absent or overrides `write_json_predictions_step.params.output_path`. In parallel mode it refuses to auto-create the writer; declare it explicitly and set `after` dependencies.

`--work-dir` is not a YAML field. It supplies `InferenceContext.work_dir` for the invocation. Relative caches and intermediate paths are interpreted by steps, with the built-in checkpoint cache resolving relative to this directory.

## Step names and registration

A step mapping is resolved through `INFERENCE_STEPS`:

```yaml
- name: resolve_checkpoints_step
  params:
    checkpoints: {model: /path/model.pt}
```

The name must be registered in the CLI process. `id` changes only the graph identifier; it does not change the registry key. `after` may be a single string or a list of strings. In sequential mode, the builder makes each step depend on the previous step and ignores explicit `after` relationships for scheduling.
