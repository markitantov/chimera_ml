# API моделей

Модели реализуют `BaseModel.forward(Batch) -> ModelOutput`. Четыре встроенные
модели слияния получают из `Batch.inputs` только тензоры признаков. Их конструкторы
объявляют `input_dims`, `shared_dim` и `output_dim`; проекции модальностей и слои
предсказания создаются внутри. Модели можно создавать напрямую в Python или через
реестр `MODELS` и YAML. См. [Модели](../user-guide/models.md).

## Контракт модели

::: chimera_ml.models.base.BaseModel

## Модели слияния

::: chimera_ml.models.tfs_fusion_model.TFSFusionModel

::: chimera_ml.models.tfs_fusion_model.tfs_fusion_model

::: chimera_ml.models.tfs_fusion_model.TFSDCMMOEFusionModel

::: chimera_ml.models.tfs_fusion_model.tfs_dcmmmoe_fusion_model

::: chimera_ml.models.raav_fusion.RAAVFusionModel

::: chimera_ml.models.raav_fusion.raav_fusion_model

::: chimera_ml.models.tomar_fusion.TOMARFusionModel

::: chimera_ml.models.tomar_fusion.tomar_fusion_model
