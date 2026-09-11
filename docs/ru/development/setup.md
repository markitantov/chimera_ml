# Development setup

Для contributor setup используйте Python 3.12 и Poetry:

~~~bash
git clone https://github.com/markitantov/chimera_ml.git
cd chimera_ml
poetry install --with dev,docs
poetry run pre-commit install
~~~

Dev group содержит pytest, coverage, Ruff, pre-commit и scikit-learn; docs group
— MkDocs, Material for MkDocs, mkdocstrings и Python handler.

~~~bash
poetry check
poetry run pre-commit run --all-files
poetry run pytest -q
~~~

Не добавляйте generated sites, local logs, credentials или model artifacts
без явной необходимости.
