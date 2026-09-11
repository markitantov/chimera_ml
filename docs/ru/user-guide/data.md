# Data и loaders

Datamodule factory возвращает object с train_dataloader, val_dataloader и
test_dataloader. Built-in DataModule создаёт PyTorch DataLoader из одного
dataset, mapping named datasets или sequence datasets.

MaskingCollate собирает tensor fields в Batch, может padding-ить sequences и
создаёт flat mask keys, например audio_mask и sequence_mask. Plugin может
зарегистрировать собственный collate в collates registry.

## Multiple loaders

Loader может быть одним DataLoader, mapping или list/tuple:

- single — использовать первый loader;
- round_robin — чередовать loaders в порядке объявления;
- weighted — выбирать loaders с train_loader_weights.

train_stop_on: min останавливается при исчерпании первого active loader; max
продолжает до исчерпания всех. Имена mapping sanitizing-ся для стабильных
metric keys.

## Batch contract

Sample должен содержать input tensors и может содержать targets, masks и meta:

~~~python
Batch(
    inputs={"audio": audio_tensor, "video": video_tensor},
    targets=target_tensor,
    masks={"audio_mask": audio_mask},
    meta={"id": sample_id},
)
~~~

Batch.get_masks() возвращает все masks, а с аргументом — один flat key. Для
совместимости читаются также legacy masks["mask"] и meta["masks"], с
преобразованием имён в modality_mask.
