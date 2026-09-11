# API models

Models реализуют \`BaseModel.forward(Batch) -> ModelOutput\`. Built-in fusion
models включают feature-level, prediction-level и gated variants. Обычно они
собираются через \`MODELS\` registry из configuration; encoders, heads и
submodels передаются как constructor dependencies. Практика:
[Models](../user-guide/models.md).

## Model contract

::: chimera_ml.models.base.BaseModel

## Feature и prediction fusion

\`FeatureFusionModel\` concatenates encoder embeddings перед head.
\`PredictionFusionModel\` объединяет per-modality predictions через mean, sum
или weighted averaging.

::: chimera_ml.models.fusion.FeatureFusionModel

::: chimera_ml.models.fusion.feature_fusion_model

::: chimera_ml.models.fusion.PredictionFusionModel

::: chimera_ml.models.fusion.prediction_fusion_model

## Gated fusion

Gated models учат per-modality weights и учитывают optional modality masks.
\`GatedFusionModel\` применяет gates к embeddings перед head;
\`GatedPredictionFusionModel\` применяет gates к logits и требует известное число
classes.

::: chimera_ml.models.gating.GatedFusionModel

::: chimera_ml.models.gating.gated_fusion_model

::: chimera_ml.models.gated_prediction_fusion.GatedPredictionFusionModel

::: chimera_ml.models.gated_prediction_fusion.gated_prediction_fusion_model
