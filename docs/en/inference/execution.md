# Sequential and parallel pipelines

## Sequential execution

When `pipeline.parallel` is omitted or `false`, the builder links steps in configuration order. The first step has no dependency; every later step depends on the previous one, so it receives all artifacts written upstream.

```yaml
pipeline:
  name: sequential-demo
  parallel: false
steps:
  - name: prepare_input_step
  - name: resolve_checkpoints_step
    params:
      checkpoints: {model: /models/model.pt}
  - name: prediction_step
    after: [ignored-for-sequential-scheduling]
    params: {checkpoint_key: model}
  - name: write_json_predictions_step
```

The `after` field is not used to reorder a sequential pipeline. The CLI `--output` can create or override the JSON writer in this mode. The writer's default path is the input path with its suffix replaced by `.json`.

## Parallel DAG execution

With `pipeline.parallel: true`, `after` is authoritative. Node IDs default to `name`; repeated registry names require explicit unique IDs. Nodes without `after` are independent roots and may run concurrently. A downstream node starts after all listed dependencies complete.

```text
prepare ─────┬──> audio_features ──┐
             └──> video_features ──┴──> predict ──> write_json
```

```yaml
pipeline:
  name: dag-demo
  parallel: true
steps:
  - name: prepare_input_step
  - id: audio
    name: extract_audio_step
    after: [prepare_input_step]
  - id: video
    name: extract_video_step
    after: [prepare_input_step]
  - name: prediction_step
    after: [audio, video]
  - name: write_json_predictions_step
    after: [prediction_step]
```

The step names above are illustrative plugin keys; verify them in your environment. The DAG mechanism itself is core behavior.

Each node receives a copy of the current artifact mapping. Only keys written with `set_artifact` are merged back. A downstream node may overwrite an upstream key when it depends on that owner; unrelated branches may not write the same key. This is artifact scheduling, not automatic batching or parallelization inside a model.

The builder/pipeline rejects missing dependencies, self-dependencies, duplicate IDs, and dependency cycles. In parallel mode, unordered conflicting writes raise an artifact ownership `ValueError`. If parallel mode has multiple nodes but no `after` fields, the builder emits a warning because all nodes are independent roots. Output writers must be declared explicitly with their dependency chain.
