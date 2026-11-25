# Checkpoints and artifacts

A model checkpoint is a file containing model state, while an inference artifact is any named value exchanged between steps. The core checkpoint step resolves files; a plugin prediction step is responsible for constructing the model, loading the state dictionary, moving it to `ctx.device`, and checking architecture compatibility.

## Resolve local or remote checkpoints

```yaml
- name: resolve_checkpoints_step
  params:
    cache_dir: model_cache
    force_download: false
    checkpoints:
      audio: /models/audio.pt
      fusion: https://example.org/models/fusion.pt
```

Local paths must exist and point to files. HTTP(S) references are downloaded to `ctx.work_dir/model_cache` by default or to the configured absolute cache directory. The generated remote filename includes a short hash of the URL. Existing cached files are reused when `force_download: false`; `true` downloads again. Downloads stream in chunks of `chunk_size` bytes (default `8192`) and use a 30-second requests timeout. A failed download removes the partial `.part` file.

The step publishes:

```text
artifacts["checkpoints"] = {"audio": "/absolute/path/audio.pt", ...}
artifacts["cache_dir"] = /absolute/path/to/cache
```

If `artifacts["checkpoints"]` already exists it must be a mapping and is extended. Missing local paths, non-file paths, invalid mappings, and failed downloads raise errors.

## Pass artifacts through the context

```python
checkpoint = ctx.get_artifact("checkpoints", {}).get("fusion")
ctx.set_artifact("features", features)
ctx.set_artifact("predictions", predictions)
output_path = ctx.get_artifact("output_path")
```

Use `get_artifact` with a default for optional values and `set_artifact` for every value that downstream steps must receive. The pipeline tracks written keys to enforce DAG ownership. `ctx.predictions` is a property alias for `get_artifact("predictions")`.

The built-in JSON writer publishes `output_path` after writing. Intermediate artifacts remain in the final context in Python API use, even when they are not serialized by the JSON writer.

## Loading compatibility

Checkpoint resolution alone does not validate the model. A plugin step should load the expected state dictionary format and raise a clear error when the model architecture, modality names, or tensor dimensions do not match. Keep checkpoint keys stable between the resolver and prediction step; for example, use `checkpoint_key: fusion` when the resolver publishes `checkpoints.fusion`.
