# Models

Model наследует BaseModel, реализует forward(batch: Batch) -> ModelOutput.
ModelOutput.preds — основные logits или regression tensor; aux может содержать
embeddings, modality predictions или features для callbacks/evaluation.

Built-in fusion factories:

| Registry key | Behavior |
| --- | --- |
| tfs_fusion_model | Направленная temporal pair fusion и иерархическая агрегация групп. |
| tfs_dcmmmoe_fusion_model | Входозависимые веса над направленными TFS pairs. |
| raav_fusion_model | Агрегация frame modalities и refinement через context latents. |
| tomar_fusion_model | Gated residual refinement anchor modality. |

Если modality отсутствует во входе, fusion model её пропускает; если не
осталось modalities, возникает ValueError. TFS, RAAV и TOMAR используют общий
`shared_dim`; RAAV требует одинаковую temporal length frame inputs, а TOMAR —
доступную anchor modality.

Для plugin model зарегистрируйте factory в MODELS и сохраняйте эту boundary.
