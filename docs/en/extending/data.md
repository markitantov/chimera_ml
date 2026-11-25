# Custom datasets and DataModules

A custom Dataset returns the mapping consumed by a registered collate. The
functions in the first example are application-specific placeholders.

~~~python
import torch
from torch.utils.data import Dataset
from chimera_ml.core import DATAMODULES
from chimera_ml.data import DataModule, MaskingCollate

class MyDataset(Dataset):
    def __init__(self, root: str, split: str):
        self.items = load_index(root, split)
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        item = self.items[index]
        return {"inputs": {"features": load_features(item)},
                "targets": load_target(item), "meta": {"id": str(index)}}

@DATAMODULES.register("my_datamodule")
def my_datamodule(*, data_root: str, batch_size: int = 32, collate_fn=None,
                  **loader_kwargs):
    return DataModule(
        train_dataset=MyDataset(data_root, "train"),
        val_dataset=MyDataset(data_root, "val"),
        batch_size=batch_size,
        collate_fn=collate_fn or MaskingCollate(),
        **loader_kwargs,
    )
~~~

load_index, load_features, and load_target must be implemented by the plugin.
Configure data.name as my_datamodule and select a built-in collate with the
top-level collate section.

## Complete synthetic batch inspection

This deterministic example is complete and does not train a model.

~~~python
import torch
from torch.utils.data import Dataset
from chimera_ml.core import DATAMODULES
from chimera_ml.data import DataModule, MaskingCollate

class SyntheticDataset(Dataset):
    def __init__(self, size: int = 4):
        self.features = torch.arange(size * 3, dtype=torch.float32).reshape(size, 3)
        self.targets = self.features.sum(dim=1, keepdim=True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index: int):
        return {"inputs": {"features": self.features[index]},
                "targets": self.targets[index],
                "meta": {"id": f"sample-{index}"}}

@DATAMODULES.register("synthetic_datamodule")
def synthetic_datamodule(batch_size: int = 2, collate_fn=None):
    dataset = SyntheticDataset()
    return DataModule(train_dataset=dataset, val_dataset=dataset,
                      batch_size=batch_size,
                      collate_fn=collate_fn or MaskingCollate())

data_module = synthetic_datamodule(batch_size=2)
batch = next(iter(data_module.train_dataloader()))
assert batch.inputs["features"].shape == (2, 3)
assert batch.targets.shape == (2, 1)
print(batch.inputs["features"], batch.targets)
~~~

Run this file after the core package is installed and before a CLI build that
uses the registry key. The decorator must execute before configuration
construction, normally through an installed plugin entry point.

~~~yaml
data:
  name: synthetic_datamodule
  params:
    batch_size: 2
collate:
  name: masking_collate
  params:
    pad_sequences: false
~~~

The YAML is a matching configuration fragment; it is not a training command.
The CLI injects the selected top-level collate as collate_fn only when the
DataModule factory accepts that keyword.
