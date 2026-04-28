# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Planned

- Built-in CLI inference pipeline command for end-to-end production-style inference (for example:
  `chimera-ml infer --config-path ... --input ... --output-dir ...`). The command should support
  plugin-defined pipeline stages such as media ingestion, audio extraction, frame extraction/downsampling,
  ASR/text extraction, modality-specific preprocessing, checkpoint loading, model inference, and structured
  prediction/artifact export.

## [0.2.2] - 2026-04-28

### Added

- Shared `training/non_finite.py` utilities for compact non-finite diagnostics in the training loop.
- Focused trainer tests covering non-finite predictions, loss, and gradients.

### Changed

- `README.md` now documents fail-fast trainer behavior for non-finite predictions, loss, gradients, and gradient norms.
- `build_from_registry(..., smart_inject=True)` docs now clarify that only explicitly declared injected parameters are passed through.

### Fixed

- Trainer now fails fast with richer debug context when predictions or loss become `NaN`/`Inf`.
- Trainer now checks unscaled gradients and clipped gradient norms for non-finite values before `optimizer.step()`.

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
