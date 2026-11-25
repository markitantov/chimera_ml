# API моделей

Models реализуют BaseModel.forward(Batch) -> ModelOutput. Четыре встроенные
fusion-модели получают feature-only tensors из Batch.inputs. Их constructors
объявляют input_dims, shared_dim и output_dim; modality projections и
prediction layers создаются внутри. Модели можно создавать напрямую в Python
или через MODELS registry и YAML. См. [Models](../user-guide/models.md).

## Контракт модели

::: chimera_ml.models.base.BaseModel

## Модели слияния

::: chimera_ml.models.tfs_fusion.TFSFusionModel

::: chimera_ml.models.tfs_fusion.tfs_fusion_model

::: chimera_ml.models.tfs_fusion.TFSDCMMOEFusionModel

::: chimera_ml.models.tfs_fusion.tfs_dcmmmoe_fusion_model

::: chimera_ml.models.raav_fusion.RAAVFusionModel

::: chimera_ml.models.raav_fusion.raav_fusion_model

::: chimera_ml.models.tomar_fusion.TOMARFusionModel

::: chimera_ml.models.tomar_fusion.tomar_fusion_model
