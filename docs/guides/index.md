# Guides

Chimera ML separates runtime infrastructure from task-specific experiment
components. The usual workflow is:

1. install `chimera-ml` and a plugin package;
2. describe data, model, loss, optimizer, metrics, logging, and callbacks in
   YAML;
3. validate the configuration with `chimera-ml validate-config`;
4. run training, evaluation, inference, or a parameter sweep with the CLI.

## Plugins and registries

Components register under names such as `models`, `losses`, `metrics`,
`callbacks`, and `inference_steps`. Plugins are discovered through the
`chimera_ml.plugins` Python entry-point group. Use these commands to inspect
the current process:

```bash
chimera-ml plugins list
chimera-ml registry list --type models
chimera-ml registry list --type inference_steps
```

## Configuration and examples

The root [README](https://github.com/markitantov/chimera_ml/blob/main/README.md)
contains the complete configuration model and CLI command reference. The
example READMEs then provide task-specific paths and dependencies:

- [VA estimation](https://github.com/markitantov/chimera_ml/blob/main/examples/va_estimation/README.md)
- [ORAGEN](https://github.com/markitantov/chimera_ml/blob/main/examples/oragen/README.md)
- [Affective states recognition](https://github.com/markitantov/chimera_ml/blob/main/examples/affective_states_recognition/README.md)

Those examples are intentionally kept next to their code and configs. This
documentation site provides the shared project concepts and navigation; it is
not a copy of every example README.
