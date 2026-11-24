# Models

A model subclasses BaseModel and implements forward(batch: Batch) ->
ModelOutput. ModelOutput.preds is the main logits or regression tensor; aux
can carry embeddings, per-modality predictions, or features used by callbacks
and feature evaluation.

The built-in fusion factories are:

| Registry key | Behavior |
| --- | --- |
| tfs_fusion_model | Performs directed temporal pair fusion and hierarchical group aggregation. |
| tfs_dcmmmoe_fusion_model | Learns input-dependent weights over directed TFS pairs. |
| raav_fusion_model | Aggregates frame modalities and refines them with context latents. |
| tomar_fusion_model | Applies gated residual refinements to an anchor modality. |

A fusion model skips a modality absent from inputs and raises ValueError when
no configured modality is present. TFS, RAAV, and TOMAR accept projected
embeddings with a shared `shared_dim`; RAAV additionally requires equal frame
sequence lengths, and TOMAR requires an available anchor modality.

For a plugin model, register a factory under MODELS and keep the boundary
stable. The [API reference](../api/models.md) contains core interfaces.
