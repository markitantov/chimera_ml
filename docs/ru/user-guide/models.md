# Models

Структурированный каталог находится на страницах [Models](../components/models.md) и [сравнение fusion](../components/fusion.md).

Model наследует BaseModel и реализует forward(batch: Batch) -> ModelOutput.
ModelOutput.preds — основной тензор предсказания, а ModelOutput.aux содержит
объединённые представления, маски валидности, веса и диагностические данные
механизмов внимания.

## Встроенные модели слияния признаков

Четыре встроенные модели получают заранее вычисленные признаки напрямую из
Batch.inputs. Они сами создают проекции модальностей и слои предсказания;
внешние словари энкодеров и головы предсказания в constructor не передаются.

| Registry key | Class | Prediction shape | Основной механизм |
| --- | --- | --- | --- |
| tfs_fusion_model | TFSFusionModel | [B, output_dim] | Иерархия направленных связей в стиле MASAI и статистическая агрегация |
| tfs_dcmmmoe_fusion_model | TFSDCMMOEFusionModel | [B, output_dim] | Иерархия TFS с адаптивными весами пар |
| raav_fusion_model | RAAVFusionModel | [B, T, output_dim] | Слияние кадров и внимание к латентным векторам контекста |
| tomar_fusion_model | TOMARFusionModel | [B, output_dim] | Остаточные поправки с условием на anchor |

Общие поля constructor — input_dims, shared_dim и output_dim. Для RAAV
обязательны frame_modalities и context_modality, для TOMAR —
anchor_modality. Подробности о масках, aux values и граничных случаях:

- [TFS](tfs-fusion.md)
- [TFS-DCMMOE](tfs-dcmmmoe-fusion.md)
- [RAAV](raav-fusion.md)
- [TOMAR](tomar-fusion.md)

Модальность может отсутствовать в Batch.inputs или быть недоступной для
образца при false маске. Для TFS и TFS-DCMMOE нужны как минимум два тензора
модальностей в Batch.inputs, чтобы сформировать пару; если все их временные
позиции замаскированы, fused становится нулевым, но предсказание всё равно
возвращается. RAAV требует одинаковую длину последовательности у присутствующих
кадровых модальностей, но временная шкала контекста может отличаться. TOMAR
требует anchor для каждого образца.

Модели можно создать через Python-классы или через MODELS registry из YAML.
[API reference](../api/models.md) содержит публичные классы и фабрики.
