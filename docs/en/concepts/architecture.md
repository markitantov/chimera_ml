# Architecture

The `src/chimera_ml` package is split by runtime responsibility:

- `core`: `Batch`, `ModelOutput`, `ExperimentConfig`, and `Registry`;
- `data`: datamodule and loader normalization/collation helpers;
- `models`, `losses`, and `metrics`: model boundary and built-in training
  primitives;
- `training`: `Trainer`, builders, optimizer/scheduler factories, and sweeps;
- `callbacks` and `logging`: lifecycle hooks, local logs, and MLflow;
- `inference`: YAML-driven steps, shared artifacts, and sequential/DAG
  execution;
- `utils`: entry-point loading, seeds, sweep helpers, and source snapshots.

The CLI composes these pieces. It does not contain task-specific datasets or
models. Those live in an installed plugin and register factories under the
same process-wide registry objects.

The model boundary is explicit: a dataloader yields `Batch`, a model accepts
`Batch` and returns `ModelOutput`, losses consume both, and metrics update from
both. This keeps a plugin independent of the CLI's command implementation.
