# Пользовательские Dataset и DataModule

Custom Dataset возвращает mapping, который принимает зарегистрированный
collate. Функции load_index, load_features и load_target в первом примере —
application-specific placeholders.

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
    return DataModule(train_dataset=MyDataset(data_root, "train"),
                      val_dataset=MyDataset(data_root, "val"),
                      batch_size=batch_size,
                      collate_fn=collate_fn or MaskingCollate(),
                      **loader_kwargs)
~~~

Эти три функции должен реализовать plugin. В YAML укажите data.name:
my_datamodule и выберите collate через top-level section collate.

## Полный synthetic batch inspection

Пример детерминированный и не обучает model.

~~~python
import torch
from torch.utils.data import Dataset
from chimera_ml.core import DATAMODULES
from chimera_ml.data import DataModule, MaskingCollate

class SyntheticDataset(Dataset):
    def __init__(self, size: int = 4):
        self.features = torch.arange(size * 3, dtype=torch.float32).reshape(size, 3)
        self.targets = self.features.sum(dim=1, keepdim=True)
    def __len__(self): return len(self.features)
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
~~~

Запустите этот Python-файл после установки core package. Decorator должен
выполниться до построения config, обычно через установленный plugin entry point.

~~~yaml
data:
  name: synthetic_datamodule
  params: {batch_size: 2}
collate:
  name: masking_collate
  params: {pad_sequences: false}
~~~

Это matching YAML fragment, а не команда обучения. CLI передаёт выбранный
top-level collate как collate_fn, если factory принимает этот keyword.
