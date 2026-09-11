# Plugin API

Plugins populate the framework registries with built-in or project-specific
components. Call \`register_all\` during application startup to load built-ins
and Python entry-point plugins. External packages declare the
\`chimera_ml.plugins\` entry-point group; each entry point may reference a
callable registration function or a module-level object. See [Plugins](../user-guide/plugins.md)
for packaging and configuration guidance.

## Built-in registration

::: chimera_ml.plugins.register_all

## Entry-point loading

::: chimera_ml.utils.entrypoints.load_entrypoint_plugins
