# Contributor architecture

Start in src/chimera_ml when tracing a change:

- core defines containers, configuration loading, and registries;
- training/builders translates named YAML sections into runtime objects;
- training/trainer owns optimization, evaluation, callbacks, and logging;
- data normalizes dataloaders and creates masks;
- inference builds and executes step graphs;
- plugins.py imports built-ins and delegates entry-point discovery.

Tests are organized by subsystem under tests/: core, data, models, losses,
metrics, callbacks, logging, inference, plugins, CLI, and training. Examples
are separately packaged projects and are part of the integration surface.

When adding a user-facing component, update its registry registration, tests,
example/config if appropriate, API reference, and both language trees.
