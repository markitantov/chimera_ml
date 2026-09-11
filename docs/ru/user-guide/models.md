# Models

Model наследует BaseModel, реализует forward(batch: Batch) -> ModelOutput.
ModelOutput.preds — основные logits или regression tensor; aux может содержать
embeddings, modality predictions или features для callbacks/evaluation.

Built-in fusion factories:

| Registry key | Behavior |
| --- | --- |
| feature_fusion_model | Encode modalities, concatenate embeddings, dropout и head. |
| prediction_fusion_model | Per-modality models и mean/sum/weighted logits. |
| gated_fusion_model | Gates над embeddings; masks отключают недоступные modalities. |
| gated_prediction_fusion_model | Gates над logits; требует num_classes. |

Если modality отсутствует во входе, fusion model её пропускает; если не
осталось modalities, возникает ValueError. Для gated_fusion_model все
projected embeddings должны иметь shared_dim; при разных dimensions используйте
set_projection.

Для plugin model зарегистрируйте factory в MODELS и сохраняйте эту boundary.
