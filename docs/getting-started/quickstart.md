# Quick start

This short walkthrough checks the installed CLI and shows where a real
experiment configuration enters the workflow.

## Check the environment

After installing `chimera-ml`, run:

```bash
chimera-ml doctor
chimera-ml --help
```

`doctor` reports basic Python, PyTorch, CUDA, MLflow, and registry/plugin
information without starting a training run.

## Validate a configuration

Install one of the repository's example plugins, then validate a configuration
before starting an experiment:

```bash
python -m pip install -e examples/va_estimation
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
```

Validation checks the YAML structure and required experiment fields. The
example configuration still needs its dataset and output paths configured for
an actual training run; see the
[VA estimation example README](https://github.com/markitantov/chimera_ml/blob/main/examples/va_estimation/README.md)
for those paths and the subsequent train/eval commands.

## Use the Python containers

The smallest library-level interaction uses the public `Batch` and
`ModelOutput` containers:

```python
import torch

from chimera_ml import Batch, ModelOutput

batch = Batch(
    inputs={"audio": torch.zeros(1, 16000)},
    targets=None,
)
output = ModelOutput(preds=torch.zeros(1, 1))

assert batch.inputs["audio"].shape == (1, 16000)
assert output.preds.shape == (1, 1)
```

Models consume `Batch` objects and return `ModelOutput` objects. The API
reference documents these containers and the registry used to connect built-in
or plugin components.
