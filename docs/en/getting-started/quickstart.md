# Quick start

This is a non-training check that can be run immediately after installing the core package. It does not require a dataset or checkpoint.

## 1. Check the installation

```bash
chimera-ml doctor
chimera-ml --help
chimera-ml registry list --type models
chimera-ml registry list --type collates
```

A clean core install lists the four built-in fusion model keys and the two built-in collates. A project plugin adds its own keys.

## 2. Load a real plugin and validate its YAML

From the repository root:

```bash
python -m pip install -e examples/va_estimation
chimera-ml plugins list
chimera-ml registry list --type datamodules
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
```

The validation command checks required framework sections. It does not check that dataset paths, feature directories, credentials, or optional model files exist, so those prerequisites still need to be prepared.

## 3. Inspect the public Python containers

```python
import torch
from chimera_ml import Batch, ModelOutput

batch = Batch(
    inputs={"audio": torch.zeros(2, 16000)},
    targets=torch.zeros(2, 1),
    masks={"audio_mask": torch.ones(2)},
    meta={"sample_meta": [{"id": "a"}, {"id": "b"}]},
)
output = ModelOutput(preds=torch.zeros(2, 1), aux={"features": torch.zeros(2, 8)})
assert batch.inputs["audio"].shape == (2, 16000)
assert output.preds.shape == (2, 1)
```

A model receives `Batch` and returns `ModelOutput`. The example above is only a container check; it does not construct a model or train anything.
