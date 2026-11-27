# Модели

Каждая модель реализует BaseModel.forward(Batch) -> ModelOutput. Встроенные
модели слияния получают признаки из Batch.inputs, создают проекции
модальностей внутри себя и возвращают предсказания в ModelOutput.preds.

| Ключ в реестре | Класс | Выход | Назначение |
| --- | --- | --- | --- |
| tfs_fusion_model | TFSFusionModel | [B, output_dim] | Иерархическое слияние направленных пар |
| raav_fusion_model | RAAVFusionModel | [B, T, output_dim] | Покадровое слияние с уточнением контекстом |
| tomar_fusion_model | TOMARFusionModel | [B, output_dim] | Слияние с опорной модальностью |

Подробности о конструкторах, масках, вспомогательных выходах и примерах:

- TFS: tfs-fusion.md
- RAAV: raav-fusion.md
- TOMAR: tomar-fusion.md

Модели можно создавать напрямую в Python или через реестр MODELS.
