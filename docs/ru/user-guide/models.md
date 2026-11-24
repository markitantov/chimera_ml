# Models

Model наследует BaseModel и реализует forward(batch: Batch) -> ModelOutput.
ModelOutput.preds — основной prediction tensor, а ModelOutput.aux содержит
fused representations, validity masks, weights и attention diagnostics.

## Built-in fusion models

Четыре встроенные fusion-модели получают precomputed features напрямую из
Batch.inputs. Они сами создают modality projections и prediction layers; внешние
encoder dictionaries и prediction heads в constructor не передаются.

| Registry key | Class | Prediction shape | Основной механизм |
| --- | --- | --- | --- |
| tfs_fusion_model | TFSFusionModel | [B, output_dim] | MASAI-style directed hierarchy и statistical pooling |
| tfs_dcmmmoe_fusion_model | TFSDCMMOEFusionModel | [B, output_dim] | TFS hierarchy с adaptive masked pair gates |
| raav_fusion_model | RAAVFusionModel | [B, T, output_dim] | Frame fusion и context-latent attention |
| tomar_fusion_model | TOMARFusionModel | [B, output_dim] | Anchor-conditioned gated residuals |

Общие поля constructor — input_dims, shared_dim и output_dim. Для RAAV
обязательны frame_modalities и context_modality, для TOMAR —
anchor_modality. Подробности о masks, aux values и edge cases:

- [TFS](tfs-fusion.md)
- [TFS-DCMMOE](tfs-dcmmmoe-fusion.md)
- [RAAV](raav-fusion.md)
- [TOMAR](tomar-fusion.md)

Modality может отсутствовать в Batch.inputs или быть недоступной для sample
при false mask. TFS и TFS-DCMMOE требуют доступную directed pair. RAAV требует
одинаковую temporal length у присутствующих frame modalities, но context
timeline может отличаться. TOMAR требует anchor для каждого sample.

Модели можно создать через Python classes или через MODELS registry из YAML.
[API reference](../api/models.md) содержит публичные classes и factories.
