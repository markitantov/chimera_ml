# API данных

Подсистема данных превращает наборы данных в контейнеры DataLoader для Trainer.
DataModule строит обучающую, валидационную и тестовую выборки. Доступны две встроенные
collate-функции: устаревшая MaskingCollate с ключом `masking_collate` и
MultimodalSequenceCollate с ключом `multimodal_sequence_collate`. Формы, маски и
примеры YAML описаны в разделе [Данные](../user-guide/data.md).

## Модуль данных

::: chimera_ml.data.datamodule.DataModule

## Устаревшее объединение образцов

::: chimera_ml.data.masking_collate.MaskingCollate

::: chimera_ml.data.masking_collate.masking_collate

## Объединение независимых последовательностей

::: chimera_ml.data.multimodal_sequence_collate.MultimodalSequenceCollate

::: chimera_ml.data.multimodal_sequence_collate.multimodal_sequence_collate

## Утилиты загрузчиков

::: chimera_ml.data.loader_utils.normalize_loaders

::: chimera_ml.data.loader_utils.sanitize_split_name
