# Custom inference steps

An inference step implements `run(ctx: InferenceContext) -> InferenceContext`, reads input/device/upstream artifacts, and publishes values with `ctx.set_artifact`. The complete deterministic implementation, pipeline configuration, DAG dependencies, and optional XAI callback are documented in [Custom prediction steps](../inference/custom-steps.md) and [XAI](../user-guide/xai.md).

For a plugin, put the factory behind the `chimera_ml.plugins` entry point exactly as shown in [Plugins](plugins.md). A registered key becomes available to a separate CLI process after installation; in-process decorator registration alone is not enough.
