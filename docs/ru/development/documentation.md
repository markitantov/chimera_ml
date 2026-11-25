# Разработка документации

Английская документация — канонический источник; русские страницы должны отражать
соответствующие английские пути после завершения английской страницы.

Локальный просмотр:

~~~bash
poetry run mkdocs serve -f mkdocs.en.yml
poetry run mkdocs serve -f mkdocs.ru.yml
~~~

Строгие сборки:

~~~bash
poetry run mkdocs build --strict -f mkdocs.en.yml
poetry run mkdocs build --strict -f mkdocs.ru.yml
~~~

Общие настройки находятся в `mkdocs.base.yml`, а навигация, имя сайта и `docs_dir`
для каждого языка — в `mkdocs.en.yml` и `mkdocs.ru.yml`. Страницы API используют
mkdocstrings, а примеры должны подтверждаться исходным кодом, тестами или поставляемым YAML.
