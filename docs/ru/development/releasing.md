# Выпуск релиза

Полный процесс описан в `RELEASING.md`:

1. обновите `CHANGELOG.md` в разделе Unreleased;
2. выполните проверки Poetry, pre-commit, тесты, сборки документации и пакета;
3. измените версию через Poetry;
4. создайте commit/push и GitHub Release или тег;
5. `publish.yml` соберёт дистрибутивы и опубликует их в PyPI через доверенную
   публикацию после успешного CI для того же commit;
6. проверьте установку в чистой среде.

~~~bash
poetry install --with dev,docs
poetry check
poetry run pre-commit run --all-files
poetry run pytest -q
poetry run mkdocs build --strict -f mkdocs.en.yml
poetry run mkdocs build --strict -f mkdocs.ru.yml
poetry build
~~~

`workflow_dispatch` с `publish=false` только собирает артефакты; публикация
требует `publish=true` и прохождения CI. Секреты не добавляйте в файлы репозитория.
