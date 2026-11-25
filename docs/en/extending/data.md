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


## Complete synthetic registration example

This deterministic example inspects a batch and does not train a model. The
load_index/load_features/load_target functions in the template above are
application placeholders.

    import torch
    from torch.utils.data import Dataset
    from chimera_ml.core import DATAMODULES
    from chimera_ml.data import DataModule, MaskingCollate

    class SyntheticDataset(Dataset):
        def __init__(self, size=4):
            self.x = torch.arange(size * 3, dtype=torch.float32).reshape(size, 3)
            self.y = self.x.sum(dim=1, keepdim=True)
        def __len__(self): return len(self.x)
        def __getitem__(self, index):
            return {"inputs": {"features": self.x[index]},
                    "targets": self.y[index], "meta": {"id": str(index)}}

    @DATAMODULES.register("synthetic_datamodule")
    def synthetic_datamodule(batch_size=2, collate_fn=None):
        dataset = SyntheticDataset()
        return DataModule(train_dataset=dataset, val_dataset=dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn or MaskingCollate())

    data_module = synthetic_datamodule(batch_size=2)
    batch = next(iter(data_module.train_dataloader()))
    assert batch.inputs["features"].shape == (2, 3)
    assert batch.targets.shape == (2, 1)

    data:
      name: synthetic_datamodule
      params: {batch_size: 2}
    collate:
      name: masking_collate
      params: {pad_sequences: false}

The decorator must run before configuration construction, normally through an
installed plugin entry point. The CLI injects a top-level collate as collate_fn
when the DataModule factory accepts that keyword.
