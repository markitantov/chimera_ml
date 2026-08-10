# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

## [0.3.0] - 2026-08-10

### Added

- `chimera-ml sweep` now supports Optuna sweeps via `method: optuna`, typed search spaces, and a config-built `sweep_target_callback` that reports the objective value after each normal training run.
- Sweep documentation now describes grid and Optuna execution, dry-run behavior, and the generated sweep artifact directory structure.

### Changed

- Simplified the CLI sweep flow into explicit `grid` and `optuna` branches while keeping trial config generation and manifest persistence in the sweep classes.

## [0.2.5.post1] - 2026-06-29

### Fixed

- `sweep` run names now honor `experiment_info.params.run_name` as the generated run-name base, avoiding duplicated trial identifiers such as `sweep-4713-001_..._sweep-4713-001`.
- `sweep` metadata directories now resolve from `console_file_logger.params.log_path` after applying the first trial overrides, so sweep manifests follow the configured log root when it is changed by the sweep config.

## [0.2.5] - 2026-06-26

### Added

- Sweep runs now keep per-series metadata under `logs/<experiment_name>/_sweeps/<sweep_id>/`, including `base_config.yaml`, `sweep_config.yaml`, `manifest.yaml`, and materialized per-trial configs.
- `chimera-ml sweep` now accepts `--sweep-name`/`-n` to add a human-readable name to sweep series and trial ids.

### Changed

- Sweep trial ids are now scoped to a unique sweep series and recorded in the manifest together with the generated run name.
- `plot_confusion_matrix_callback` now logs confusion matrix artifacts as vector PDF files instead of PNG files.

## [0.2.4] - 2026-05-21

### Added

- New `examples/affective_states_recognition` plugin package with multimodal emotion and sentiment recognition pipelines over audio, video, and text. It includes training/evaluation configs, an inference config, plugin source code, and usage docs for CMU-MOSEI, MELD, and RAMAS-oriented workflows.

### Changed

- Parallel inference artifact merging now tracks artifact keys written through `InferenceContext.set_artifact(...)` instead of inferring updates by deep-comparing full artifact snapshots. This makes DAG execution more robust around shared upstream artifacts while preserving explicit overwrite checks between unrelated branches.

### Fixed

- CLI `inference` now rejects `--output/-o` auto-creation of `write_json_predictions_step` when `pipeline.parallel: true` is enabled; the output step must be declared explicitly in the config together with its `after` dependencies.

## [0.2.3] - 2026-05-04

### Added

- Built-in CLI `inference` flow based on a shared `InferenceContext` and registry-driven inference steps.
- Inference pipeline support for both sequential execution and DAG/parallel execution with explicit `after` dependencies.
- Built-in inference output steps for JSON printing and JSON file export.
- Built-in `resolve_checkpoints_step` for resolving local or remote checkpoints into local cached files and exposing them via `artifacts["checkpoints"]`.
- New `examples/oragen` plugin package with audio and multimodal ORAGEN training/inference pipelines, configs, export helper script, and documentation.
- Automated tests covering inference pipeline execution, DAG behavior, CLI wiring, and checkpoint resolution.

## [0.2.2] - 2026-04-28

### Added

- Focused trainer tests for current trainer behavior.

### Changed

- `build_from_registry(..., smart_inject=True)` docs now clarify that only explicitly declared injected parameters are passed through.

### Fixed

- Trainer/runtime edge cases around factory injection and evaluation flow.

## [0.2.1] - 2026-04-24

### Added

- Per-run `BuildContext` shared across CLI build stages so plugin components can exchange runtime metadata without duplicating YAML config.
- `BuildContext` support in registry builders for datamodules, models, losses, metrics, optimizers, schedulers, callbacks, collates, and loggers.
- Context registration hooks via `BuildContext.register(...)` and `BuildContext.register_many(...)` for components that implement `describe_context(...)`.

### Changed

- CLI `train` and `eval` flows now create and propagate a shared build context before constructing downstream components.
- Plugin authoring flow now favors explicit context-based metadata exchange over ad hoc runtime config mutation.
- `README.md` documentation now includes guidance for using `BuildContext` in plugin factories and components.

## [0.2.0] - 2026-04-22

### Added

- Built-in CLI sweep command for hyperparameter trial series (for example:
  `chimera-ml sweep --base-config ... --sweep-config ...`).

## [0.1.0] - 2026-04-02

### Added

- Initial public package structure for `chimera-ml` (`src/` layout) with core training primitives, registries, callbacks, logging, losses, metrics, and fusion models.
- Data pipeline utilities: generic datamodule, loader helpers, masking collate, and mixed-loader training support.
- New CLI commands: `validate-config`, `doctor`, `registry list`, `plugins list`.
- New confusion-matrix plotting callback (`plot_confusion_matrix_callback`) and cached split-output storage used by prediction collection.
- `examples/va_estimation` plugin package with multimodal and audio pipelines, configs, and training entrypoints.
- Expanded automated tests across callbacks, CLI, core, data, losses, metrics, and trainer integrations.
- Automated PyPI release workflow (`.github/workflows/publish.yml`) with manual trigger support.
- CI gate in publish workflow: PyPI publish is blocked unless `CI` succeeded for the same commit.
- `RELEASING.md` with release and post-publish verification checklist.
- Expanded dependency set for `examples/va_estimation`.

### Changed

- Metrics subsystem migrated away from sklearn-based metrics to internal PRF/regression/confusion-matrix implementations.
- Renamed `mlflow_predictions_callback` to `collect_predictions_callback`.
- Trainer internals refactored for callback/metrics orchestration and prediction caching.
- Config/training plumbing refactored (`yaml_config` removed in favor of the current config flow).
- Documentation refreshed (`README.md`, `CONTRIBUTING.md`, `RELEASING.md`) to match the current CLI and release flow.
- CI now runs `poetry check`, `pre-commit`, `pytest -q`, and `poetry build`.
- Linting policy tightened: line length limit set to 120.
- Linting policy tightened: `E501` enabled.
- Linting policy tightened: additional rule families enabled (`C4`, `PIE`, `RET`, `RUF`, `SIM`).
- Linting now covers `examples/` again.
- Local machine-specific config folders are now ignored via gitignore rules.

### Fixed

- Callback reliability fixes, including snapshot callback behavior.
- Trainer/CLI stability fixes in callback+metrics integration flows.
- Trainer logging for multioutput metrics.
- `examples/va_estimation` regressions in datamodules/callbacks.
- Syntax error in `tests/training/test_trainer_smoke.py`.
- Remaining line-length violations and style issues across `src/`, `tests/`, and `examples/`.
