# Documentation development

English — canonical source; Russian pages должны зеркалить relevant English
paths после завершения English страницы.

Локальный serve:

~~~bash
poetry run mkdocs serve -f mkdocs.en.yml
poetry run mkdocs serve -f mkdocs.ru.yml
~~~

Strict builds:

~~~bash
poetry run mkdocs build --strict -f mkdocs.en.yml
poetry run mkdocs build --strict -f mkdocs.ru.yml
~~~

Общие settings находятся в mkdocs.base.yml, language-specific nav/site name/
docs_dir — в mkdocs.en.yml и mkdocs.ru.yml. API pages используют mkdocstrings,
а примеры должны быть подтверждены source, tests или shipped YAML.
