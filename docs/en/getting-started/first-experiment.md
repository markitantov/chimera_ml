# First experiment

This walkthrough uses `examples/va_estimation`, because it is a real plugin
package in this repository. It is a template for the framework workflow, not a
zero-download demo: its configs require the VA datasets, features, and paths
listed in the example README.

## Project shape

The important pieces are:

```text
examples/va_estimation/
├── configs/
│   ├── multimodal_train.yaml
│   └── multimodal_test.yaml
├── src/
│   └── chimera_plugin.py
└── pyproject.toml
```

`pyproject.toml` exposes the plugin as `va_estimation = "chimera_plugin:register"`.
The register function imports the modules that populate Chimera ML registries.

## Configure

Install the plugin and edit the dataset, feature, annotation, and optional
notification paths in the config:

```bash
python -m pip install -e examples/va_estimation
chimera-ml validate-config \
  --config-path examples/va_estimation/configs/multimodal_train.yaml
```

An experiment config has named components such as `data`, `model`, `loss`, and
`optimizer`, each using `name` plus `params`. `metrics`, `callbacks`, and
`logging` are lists of the same shape. See [Configuration](../concepts/configuration.md)
for the contract.

## Train

```bash
chimera-ml train \
  --config-path examples/va_estimation/configs/multimodal_train.yaml
```

The CLI creates a per-run `BuildContext`, builds each component from its
registry, and calls `Trainer.fit`. The checkpoint and snapshot callbacks write
under the configured log root. The exact output name is controlled by callback
parameters and the generated run name.

## Evaluate

```bash
chimera-ml eval \
  --config-path examples/va_estimation/configs/multimodal_test.yaml \
  --checkpoint-path path/to/checkpoint.pt
```

Evaluation loads either a checkpoint containing `model_state_dict` or a raw
state dictionary, then evaluates the merged train/validation/test loader
splits that the datamodule exposes. The checkpoint must match the model
strictly.

## Next steps

- Configure [training](../user-guide/training.md), [metrics](../user-guide/metrics.md),
  and [callbacks](../user-guide/callbacks.md).
- See the [ORAGEN tutorial](../tutorials/oragen.md) for inference and
  checkpoint caching.
- See [Writing a plugin](../user-guide/plugins.md) and the
  [plugin authoring tutorial](../development/plugins.md) to add components.
