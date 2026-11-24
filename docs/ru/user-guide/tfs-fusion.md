# TFS fusion

`tfs_fusion_model` строит общие представления modalities для входов `[B, D]`
и temporal-входов `[B, T, D]`, затем вычисляет направленные attention pairs.
Pair outputs агрегируются внутри optional groups, проходят group-specific
linear layers и передаются в head.

Модель использует `<modality>_mask` из `Batch.masks`, пропускает отсутствующие
modalities и требует минимум две доступные modality. `input_dims` задаёт
проекции encoder outputs в `shared_dim`.
