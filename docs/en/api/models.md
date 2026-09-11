# Models API

Models implement \`BaseModel.forward(Batch) -> ModelOutput\`. The built-in
fusion models cover feature-level fusion, prediction-level fusion, and gated
variants. They are commonly assembled through the \`MODELS\` registry from
configuration; encoders, heads, and submodels are supplied as constructor
dependencies. See [Models](../user-guide/models.md) for task-oriented setup.

## Model contract

::: chimera_ml.models.base.BaseModel

## Feature and prediction fusion

\`FeatureFusionModel\` concatenates encoder embeddings before the head.
\`PredictionFusionModel\` combines per-modality predictions using mean, sum, or
weighted averaging.

::: chimera_ml.models.fusion.FeatureFusionModel

::: chimera_ml.models.fusion.feature_fusion_model

::: chimera_ml.models.fusion.PredictionFusionModel

::: chimera_ml.models.fusion.prediction_fusion_model

## Gated fusion

The gated models learn per-modality weights and honor optional modality masks.
\`GatedFusionModel\` gates embeddings before the head; \`GatedPredictionFusionModel\`
gates logits and requires a known class count.

::: chimera_ml.models.gating.GatedFusionModel

::: chimera_ml.models.gating.gated_fusion_model

::: chimera_ml.models.gated_prediction_fusion.GatedPredictionFusionModel

::: chimera_ml.models.gated_prediction_fusion.gated_prediction_fusion_model
