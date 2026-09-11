# CLI reference

Executable — chimera-ml; alias chimera_ml указывает на то же Typer application.
Проверьте установленную версию командой chimera-ml --help.

## validate-config

~~~text
chimera-ml validate-config --config-path CONFIG_PATH
chimera-ml validate-config -c CONFIG_PATH [--require-experiment-name | --no-require-experiment-name]
~~~

Проверяет framework-level experiment YAML без запуска training. По умолчанию
требует experiment_info.params.experiment_name.

## doctor

~~~text
chimera-ml doctor [--plugin-group GROUP]
~~~

Печатает Python, platform, PyTorch, CUDA, MLflow, registry и entry-point
diagnostics. Default group — chimera_ml.plugins.

## train

~~~text
chimera-ml train --config-path CONFIG_PATH
chimera-ml train -c CONFIG_PATH
~~~

Строит components из registries и вызывает Trainer.fit.

## eval

~~~text
chimera-ml eval --config-path CONFIG_PATH --checkpoint-path CHECKPOINT_PATH [--with-features]
~~~

Оба пути обязательны. --with-features просит Trainer.evaluate cache features.

~~~bash
chimera-ml eval -c test.yaml --checkpoint-path checkpoints/last.pt
~~~

## inference

~~~text
chimera-ml inference --input INPUT_PATH [--output OUTPUT_PATH] --config-path CONFIG_PATH [--device auto|cpu|cuda] [--work-dir WORK_DIR]
~~~

Short options: -i, -o, -c. input и config-path обязательны. output может
создать или переопределить write_json_predictions_step только в sequential
pipeline; parallel pipeline должен объявить step в YAML.

## sweep

~~~text
chimera-ml sweep --base-config BASE_CONFIG --sweep-config SWEEP_CONFIG [--sweep-name NAME] [--max-trials N] [--dry-run]
~~~

Short options: -b, -s, -n. max-trials должен быть >= 1. Подробности в
[секции sweeps](../user-guide/sweeps.md).

## registry list

~~~text
chimera-ml registry list [--type TYPE]
~~~

TYPE: datamodules, models, losses, metrics, optimizers, schedulers, callbacks,
collates, loggers или inference_steps.

## plugins list

~~~text
chimera-ml plugins list [--group GROUP]
~~~

Default group — chimera_ml.plugins. Вложенные команды имеют форму
command subcommand.
