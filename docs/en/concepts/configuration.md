# Configuration-driven architecture

Training and evaluation use a YAML mapping. The framework validates the
top-level shape, then builders resolve each `name` through a registry and pass
`params` to the factory.

The smallest structural example is:

```yaml
seed: 0

experiment_info:
  params:
    experiment_name: my-experiment

data:
  name: my_datamodule
  params: {}

model:
  name: my_model
  params: {}

train:
  params:
    epochs: 10

loss:
  name: mse_loss
  params: {}

optimizer:
  name: adamw_optimizer
  params:
    lr: 0.001
```


## Fusion and sequence-collate example

The same registry contract can build a feature-only fusion model and an
independent sequence collate:

~~~yaml
model:
  name: raav_fusion_model
  params:
    input_dims: {video: 512, audio: 768, text: 1024}
    shared_dim: 256
    output_dim: 2
    frame_modalities: [video, audio]
    context_modality: text

collate:
  name: multimodal_sequence_collate
  params:
    pad_sequences: true
~~~

The collate emits modality-specific presence and temporal masks. Fusion models
read those masks from Batch and do not align independent timelines automatically.

When this top-level section is present, CLI `train` and `eval` resolve it with
`build_collate()` and inject the callable into the datamodule factory as
`collate_fn`. The factory must explicitly declare that keyword parameter. With
no `collate:` section, the CLI preserves the datamodule's existing default and
injects nothing; configuring both top-level `collate:` and
`data.params.collate_fn` is an error.

## Structural rules

`data`, `model`, `loss`, and `optimizer` must be mappings with non-empty
`name` values. `train` must be a mapping, and its `params` are passed to
`TrainConfig`. `scheduler` is optional and must have a `name` when present.
`metrics`, `callbacks`, and `logging` are optional lists; each list item must
have a `name`.

`experiment_info.params.experiment_name` is required by `train` and by
`validate-config` unless `--no-require-experiment-name` is used.

## Named lists and runtime injection

The runtime accepts named sections in either mapping or list form. Logging is
normally a list because multiple loggers can coexist:

```yaml
logging:
  - name: console_file_logger
    params:
      log_path: logs
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
```

Builders may inject runtime objects. For example, the optimizer factory opts
into `model`, the scheduler receives `optimizer`, and a factory declaring
`context` can receive the current `BuildContext`. With `smart_inject=True`,
only explicitly declared parameters are injected; accepting `**kwargs` alone
does not opt a factory in.

## Validation versus execution

`validate-config` checks the framework's structural contract only. Unknown
registry keys, missing files, incompatible plugin parameters, and dataset
errors appear when the corresponding workflow builds or runs the component.

For supported fields and defaults, see [Configuration reference](../user-guide/configuration.md).
