# Testing

Run all tests with:

~~~bash
poetry run pytest -q
~~~

The test suite covers registry/config behavior, loader and masking semantics,
trainer integrations, non-finite guards, callbacks, logging, inference graphs
and checkpoint resolution, CLI wiring, plugin discovery, and grid/Optuna
sweep behavior.

For a focused change, run its subsystem first:

~~~bash
poetry run pytest tests/inference tests/cli -q
poetry run pytest tests/training/test_sweep.py -q
~~~

Behavior changes need tests. Documentation-only changes still need both strict
MkDocs builds because mkdocstrings imports the package and resolves references.
