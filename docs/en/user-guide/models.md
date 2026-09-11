# Models

A model subclasses BaseModel and implements forward(batch: Batch) ->
ModelOutput. ModelOutput.preds is the main logits or regression tensor; aux
can carry embeddings, per-modality predictions, or features used by callbacks
and feature evaluation.

The built-in fusion factories are:

| Registry key | Behavior |
| --- | --- |
| feature_fusion_model | Encodes available modalities, concatenates embeddings, applies optional dropout, and calls a head. |
| prediction_fusion_model | Runs per-modality submodels and combines logits with mean, sum, or weighted averaging. |
| gated_fusion_model | Computes modality gates over embeddings; masks suppress unavailable modalities. |
| gated_prediction_fusion_model | Computes gates over per-modality logits and requires num_classes. |

A fusion model skips a modality absent from inputs and raises ValueError when
no configured modality is present. gated_fusion_model requires every projected
embedding to have shared_dim; use set_projection when encoders have different
dimensions.

For a plugin model, register a factory under MODELS and keep the boundary
stable. The [API reference](../api/models.md) contains core interfaces.
