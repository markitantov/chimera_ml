# Установка

## Требования

Текущая версия Chimera ML поддерживает Python `>=3.12,<3.13`. Пакет объявляет
PyTorch `>=2.2,<3.0` и остальные runtime dependencies в `pyproject.toml`.

## Установка из PyPI

```bash
python -m pip install chimera-ml
```

Пакет устанавливает CLI `chimera-ml`. Также доступен вариант с подчёркиванием
`chimera_ml`, но в документации последовательно используется имя с дефисом.

## Установка из исходников

Для разработки установите проект и инструменты разработки с помощью Poetry:

```bash
poetry install --with dev,docs
```

Группа `docs` содержит MkDocs, Material for MkDocs и Python handler для
`mkdocstrings`. Она намеренно отделена от production runtime dependencies.

## Установка plugin package для примера

Datamodule, модели, функции потерь, метрики и колбэки, специфичные для задачи,
обычно предоставляются внешними plugins. Из корня репозитория пример plugin
package можно установить в editable mode:

```bash
python -m pip install -e examples/va_estimation
```

Зависимости plugin package и пути к датасетам описаны в README каждого примера.
Для диагностики core CLI или автоматически сгенерированного API reference
установка плагина не требуется.
