# Models API

Models implement BaseModel.forward(Batch) -> ModelOutput. The four built-in
fusion models consume feature-only tensors from Batch.inputs. Their constructors
declare input_dims, shared_dim, and output_dim; modality projections and
prediction layers are internal. They can be constructed directly in Python or
through the MODELS registry and YAML. See [Models](../user-guide/models.md) for
task-oriented examples.

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
