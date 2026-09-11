# Releasing

Полный процесс поддерживается в RELEASING.md:

1. обновите CHANGELOG.md в Unreleased;
2. пройдите Poetry, pre-commit, tests, docs builds и package build;
3. измените version через Poetry;
4. commit/push и создайте GitHub Release или tag;
5. publish.yml соберёт distributions и опубликует PyPI через trusted
   publishing после успешного CI для того же commit;
6. проверьте install в clean environment.

~~~bash
poetry install --with dev,docs
poetry check
poetry run pre-commit run --all-files
poetry run pytest -q
poetry run mkdocs build --strict -f mkdocs.en.yml
poetry run mkdocs build --strict -f mkdocs.ru.yml
poetry build
~~~

workflow_dispatch с publish=false только собирает artifacts; публикация
требует publish=true и CI gate. Secrets не добавляйте в repository files.
