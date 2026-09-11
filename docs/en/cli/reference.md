# CLI reference

The installed executable is chimera-ml. The underscore alias chimera_ml points to
the same Typer application. Run chimera-ml --help for the version installed in
your environment.

## validate-config

Validate the framework-level experiment YAML without starting training.

~~~text
chimera-ml validate-config --config-path CONFIG_PATH
chimera-ml validate-config -c CONFIG_PATH [--require-experiment-name | --no-require-experiment-name]
~~~

The required option is --config-path/-c. The experiment-name requirement is
enabled by default and can be disabled for structural checks.

~~~bash
chimera-ml validate-config -c config.yaml
~~~

## doctor

Print Python, platform, PyTorch, CUDA, MLflow, registry, and entry-point
diagnostics.

~~~text
chimera-ml doctor [--plugin-group GROUP]
~~~

The default group is chimera_ml.plugins.

~~~bash
chimera-ml doctor
~~~

## train

Run the normal training flow from YAML.

~~~text
chimera-ml train --config-path CONFIG_PATH
chimera-ml train -c CONFIG_PATH
~~~

The option is required. Components are built from registries and the command
calls Trainer.fit.

## eval

Evaluate a checkpoint with the configured data/model/metrics/callbacks.

~~~text
chimera-ml eval --config-path CONFIG_PATH --checkpoint-path CHECKPOINT_PATH [--with-features]
~~~

Both --config-path/-c and --checkpoint-path are required. --with-features asks
Trainer.evaluate to cache model features.

~~~bash
chimera-ml eval -c test.yaml --checkpoint-path checkpoints/last.pt
~~~

## inference

Run a registry-driven pipeline over one input file.

~~~text
chimera-ml inference
  --input INPUT_PATH
  [--output OUTPUT_PATH]
  --config-path CONFIG_PATH
  [--device auto|cpu|cuda]
  [--work-dir WORK_DIR]
~~~

Short options are -i, -o, and -c. input and config-path are required. output
overrides or creates write_json_predictions_step only for sequential pipelines.
Parallel pipelines must declare that step in YAML.

~~~bash
chimera-ml inference -i video.mp4 -o prediction.json -c inference.yaml
~~~

## sweep

Run grid or Optuna trials based on a base experiment config.

~~~text
chimera-ml sweep
  --base-config BASE_CONFIG
  --sweep-config SWEEP_CONFIG
  [--sweep-name NAME]
  [--max-trials N]
  [--dry-run]
~~~

Short options are -b, -s, and -n. base-config and sweep-config are required.
max-trials must be at least 1 and limits generated/executed trials.

~~~bash
chimera-ml sweep -b config.yaml -s sweep.yaml --dry-run
~~~

See [Sweeps](../user-guide/sweeps.md) for grid syntax, typed Optuna spaces,
objectives, trial ids, and manifests.

## registry list

List registered component keys.

~~~text
chimera-ml registry list [--type TYPE]
~~~

Valid types are datamodules, models, losses, metrics, optimizers, schedulers,
callbacks, collates, loggers, and inference_steps.

~~~bash
chimera-ml registry list --type models
~~~

## plugins list

List discovered entry points.

~~~text
chimera-ml plugins list [--group GROUP]
~~~

The default group is chimera_ml.plugins.

~~~bash
chimera-ml plugins list
~~~

Nested commands use syntax command subcommand; for example registry list and
plugins list.
