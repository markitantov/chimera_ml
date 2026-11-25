# Пользовательские шаги инференса

Inference step реализует `run(ctx: InferenceContext) -> InferenceContext`, читает input/device/upstream artifacts и публикует values через `ctx.set_artifact`. Полный deterministic implementation, pipeline/DAG и XAI callback находятся в [Custom prediction steps](../inference/custom-steps.md) и [XAI](../user-guide/xai.md).

Для plugin используйте entry point `chimera_ml.plugins`, описанный в [Plugins](plugins.md). После установки key доступен отдельному CLI process; in-process decorator недостаточен.
