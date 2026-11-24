# Models API

Models implement \`BaseModel.forward(Batch) -> ModelOutput\`. The built-in
fusion models cover hierarchical pair, adaptive pair, asymmetric frame, and
anchor-residual fusion. They are commonly assembled through the \`MODELS\`
registry from configuration; encoders and heads are supplied as constructor
dependencies. See [Models](../user-guide/models.md) for task-oriented setup.

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
