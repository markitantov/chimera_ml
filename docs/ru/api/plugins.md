# API плагинов

Плагины заполняют реестры фреймворка встроенными или специфичными для проекта компонентами.
Вызовите \`register_all\` на startup приложения, чтобы загрузить built-ins и
Плагины с точками входа Python. Внешний пакет объявляет группу точек входа
\`chimera_ml.plugins\`; entry point может указывать callable registration function
или module-level object. Packaging и configuration описаны в
[Plugins](../user-guide/plugins.md).

## Встроенная регистрация

::: chimera_ml.plugins.register_all

## Загрузка через точки входа

::: chimera_ml.utils.entrypoints.load_entrypoint_plugins
