# Custom datasets and DataModules

A custom dataset returns the mapping consumed by a registered collate:

```python
import torch
from torch.utils.data import Dataset
from chimera_ml.core import DATAMODULES
from chimera_ml.data import DataModule, MaskingCollate

class MyDataset(Dataset):
    def __init__(self, root: str, split: str): self.items = load_index(root, split)
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        return {"inputs": {"features": load_features(self.items[index])}, "targets": load_target(self.items[index]), "meta": {"id": str(index)}}

@DATAMODULES.register("my_datamodule")
def my_datamodule(*, data_root: str, batch_size: int = 32, collate_fn=None, **loader_kwargs):
    return DataModule(
        train_dataset=MyDataset(data_root, "train"),
        val_dataset=MyDataset(data_root, "val"),
        batch_size=batch_size,
        collate_fn=collate_fn or MaskingCollate(),
        **loader_kwargs,
    )
```

`load_index`, `load_features`, and `load_target` are task-specific functions implemented by the plugin. Configure `data.name: my_datamodule`; select a built-in collate with top-level `collate`. Keep modality names, shapes, labels, and mask meaning stable between dataset and model.
