# Documentation development

English is the canonical source. Russian pages mirror relevant English paths
after the English page is complete.

The two sites use:

~~~bash
poetry run mkdocs serve -f mkdocs.en.yml
poetry run mkdocs serve -f mkdocs.ru.yml
~~~

Build both sites strictly:

~~~bash
poetry run mkdocs build --strict -f mkdocs.en.yml
poetry run mkdocs build --strict -f mkdocs.ru.yml
~~~

The shared settings live in mkdocs.base.yml; language-specific nav, site name,
and docs_dir live in mkdocs.en.yml and mkdocs.ru.yml. API pages use mkdocstrings
directives instead of copied docstrings. Keep examples tied to source,
tests, or shipped YAML and use relative links between pages.
