# API models

Models реализуют \`BaseModel.forward(Batch) -> ModelOutput\`. Built-in fusion
models включают hierarchical pair, adaptive pair, asymmetric frame и
anchor-residual variants. Обычно они собираются через \`MODELS\` registry из
configuration; encoders и heads передаются как constructor dependencies.

## Model contract

::: chimera_ml.models.base.BaseModel

## Fusion models

::: chimera_ml.models.tfs_fusion.TFSFusionModel

::: chimera_ml.models.tfs_fusion.tfs_fusion_model

::: chimera_ml.models.tfs_fusion.TFSDCMMOEFusionModel

::: chimera_ml.models.tfs_fusion.tfs_dcmmmoe_fusion_model

::: chimera_ml.models.raav_fusion.RAAVFusionModel

::: chimera_ml.models.raav_fusion.raav_fusion_model

::: chimera_ml.models.tomar_fusion.TOMARFusionModel

::: chimera_ml.models.tomar_fusion.tomar_fusion_model
