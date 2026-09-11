# Releasing

The public release process is maintained in RELEASING.md. The short sequence is:

1. update CHANGELOG.md under Unreleased;
2. run Poetry checks, pre-commit, tests, documentation builds, and package build;
3. bump the version with Poetry;
4. commit and push, then create a GitHub Release or tag;
5. let publish.yml build distributions and publish to PyPI through trusted
   publishing after CI has succeeded for the same commit;
6. install the published package in a clean environment and run chimera-ml --help.

Commands:

~~~bash
poetry install --with dev,docs
poetry check
poetry run pre-commit run --all-files
poetry run pytest -q
poetry run mkdocs build --strict -f mkdocs.en.yml
poetry run mkdocs build --strict -f mkdocs.ru.yml
poetry build
~~~

The workflow also supports workflow_dispatch. A manual run with publish=false
builds and validates artifacts; publishing requires publish=true and the CI
success gate. Do not put PyPI tokens or other secrets in repository files.
