# Настройка разработки

Для настройки среды разработчика используйте Python 3.12 и Poetry:

~~~bash
git clone https://github.com/markitantov/chimera_ml.git
cd chimera_ml
poetry install --with dev,docs
poetry run pre-commit install
~~~

Группа `dev` содержит pytest, coverage, Ruff, pre-commit и scikit-learn; группа `docs`
— MkDocs, Material for MkDocs, mkdocstrings и обработчик Python.

~~~bash
poetry check
poetry run pre-commit run --all-files
poetry run pytest -q
~~~

Не добавляйте сгенерированные сайты, локальные журналы, учётные данные или артефакты
моделей без явной необходимости.
