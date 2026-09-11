# Development setup

Clone the repository, select Python 3.12, and install all development groups:

~~~bash
git clone https://github.com/markitantov/chimera_ml.git
cd chimera_ml
poetry install --with dev,docs
poetry run pre-commit install
~~~

The dev group provides pytest, coverage, Ruff, pre-commit, and scikit-learn.
The docs group provides MkDocs, Material for MkDocs, mkdocstrings, and its
Python handler.

Run the core checks before a pull request:

~~~bash
poetry check
poetry run pre-commit run --all-files
poetry run pytest -q
~~~

Do not commit generated sites, local experiment logs, credentials, or model
artifacts unless a change explicitly requires them.
