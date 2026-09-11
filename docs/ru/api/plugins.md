# API plugins

Plugins заполняют framework registries built-in или project-specific components.
Вызовите \`register_all\` на startup приложения, чтобы загрузить built-ins и
Python entry-point plugins. Внешний пакет объявляет entry-point group
\`chimera_ml.plugins\`; entry point может указывать callable registration function
или module-level object. Packaging и configuration описаны в
[Plugins](../user-guide/plugins.md).

## Built-in registration

::: chimera_ml.plugins.register_all

## Entry-point loading

::: chimera_ml.utils.entrypoints.load_entrypoint_plugins
