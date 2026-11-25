# Custom datasets и DataModules

Dataset возвращает mapping, который понимает выбранный collate:

```python
from torch.utils.data import Dataset
from chimera_ml.core import DATAMODULES
from chimera_ml.data import DataModule, MaskingCollate

class MyDataset(Dataset):
    def __init__(self, root: str, split: str): self.items = load_index(root, split)
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        item = self.items[index]
        return {"inputs": {"features": load_features(item)}, "targets": load_target(item), "meta": {"id": str(index)}}

@DATAMODULES.register("my_datamodule")
def my_datamodule(*, data_root: str, batch_size: int = 32, collate_fn=None, **loader_kwargs):
    return DataModule(train_dataset=MyDataset(data_root, "train"), val_dataset=MyDataset(data_root, "val"), batch_size=batch_size, collate_fn=collate_fn or MaskingCollate(), **loader_kwargs)
```

`load_index`, `load_features`, `load_target` — task-specific functions plugin. После регистрации укажите `data.name: my_datamodule` и top-level `collate`; modality names, shapes, labels и masks должны совпадать с model contract.
