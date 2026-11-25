# Пользовательские Dataset и DataModule

Пользовательский Dataset возвращает отображение, которое принимает зарегистрированная
collate-функция. Функции `load_index`, `load_features` и `load_target` в первом
примере являются заполнителями, зависящими от приложения.

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

Эти три функции должен реализовать плагин. В YAML укажите `data.name:
my_datamodule` и выберите collate-функцию через верхнеуровневую секцию `collate`.

## Полная проверка синтетического батча

Пример детерминированный и не обучает модель.

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

Запустите этот Python-файл после установки пакета ядра. Декоратор должен
сработать до построения конфигурации, обычно через установленную точку входа плагина.

~~~yaml
data:
  name: synthetic_datamodule
  params: {batch_size: 2}
collate:
  name: masking_collate
  params: {pad_sequences: false}
~~~

Это соответствующий фрагмент YAML, а не команда обучения. CLI передаёт выбранную
верхнеуровневую collate-функцию как `collate_fn`, если функция построения принимает
этот именованный параметр.
