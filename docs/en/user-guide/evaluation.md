# Evaluation

Evaluate a checkpoint without optimization:

~~~bash
chimera-ml eval \
  --config-path config.yaml \
  --checkpoint-path logs/experiment/run/checkpoints/last.pt
~~~

The command builds the same data/model/loss/metrics/callback components as
training. It loads the checkpoint with weights_only=True on CPU and accepts
either a mapping containing model_state_dict or a raw state dictionary.
load_state_dict(strict=True) means the checkpoint architecture must match the
configured model.

The datamodule train, validation, and test loaders are flattened into split
names. A named loader retains its name when it begins with the split prefix;
otherwise the CLI prefixes it with train_, val_, or test_.

Evaluation runs one epoch with Trainer.evaluate. It calls callbacks, updates
metrics, and can cache predictions. --with-features asks the trainer to cache
features from ModelOutput.aux["features"]; it fails if the model does not
provide them and no feature extractor was supplied by direct API use.
