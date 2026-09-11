# API reference

Этот справочник объясняет контракты, которые используют подсистемы framework.
Документация объектов генерируется из установленного пакета через mkdocstrings,
а каждая страница дополнительно содержит contextual описание, связи между
configuration и objects и ссылки на task-oriented инструкции.

Начните с [Core](core.md), где описаны Batch, ModelOutput, ExperimentConfig и
Registry. Затем выберите страницу подсистемы, компонент которой вы создаёте
или настраиваете.

## Подсистемы

- [Core](core.md): общие containers, configuration и registry primitives.
- [Data](data.md): DataModule, collation, masks и loader normalization.
- [Models](models.md): model contract и multimodal fusion implementations.
- [Training](training.md): Trainer, builders, BuildContext и sweeps.
- [Inference](inference.md): contexts, DAG pipelines, steps и builders.
- [Callbacks](callbacks.md): lifecycle extensions и artifact callbacks.
- [Losses](losses.md): optimization objectives и registry factories.
- [Metrics](metrics.md): stateful epoch metrics и aggregation.
- [Logging](logging.md): console/file и MLflow logger contracts.
- [Plugins](plugins.md): built-in и entry-point registration.

Используйте [User Guide](../user-guide/index.md), когда нужно выполнить
задачу; этот раздел нужен для точных signatures, полей, return values и
extension contracts.
