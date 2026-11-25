# Справочник CLI

Исполняемый файл — `chimera-ml`; псевдоним `chimera_ml` указывает на то же приложение Typer.
Проверьте установленную версию командой `chimera-ml --help`.

## validate-config

~~~text
chimera-ml validate-config --config-path CONFIG_PATH
chimera-ml validate-config -c CONFIG_PATH [--require-experiment-name | --no-require-experiment-name]
~~~

Проверяет YAML эксперимента на уровне фреймворка без запуска обучения. По умолчанию
требуется `experiment_info.params.experiment_name`.

## doctor

~~~text
chimera-ml doctor [--plugin-group GROUP]
~~~

Печатает сведения о Python, платформе, PyTorch, CUDA, MLflow, реестрах и точках
входа. Группа по умолчанию — `chimera_ml.plugins`.

## train

~~~text
chimera-ml train --config-path CONFIG_PATH
chimera-ml train -c CONFIG_PATH
~~~

Строит компоненты из реестров и вызывает `Trainer.fit`.

## eval

~~~text
chimera-ml eval --config-path CONFIG_PATH --checkpoint-path CHECKPOINT_PATH [--with-features]
~~~

Оба пути обязательны. `--with-features` запрашивает у `Trainer.evaluate` кэширование признаков.

~~~bash
chimera-ml eval -c test.yaml --checkpoint-path checkpoints/last.pt
~~~

## inference

~~~text
chimera-ml inference --input INPUT_PATH [--output OUTPUT_PATH] --config-path CONFIG_PATH [--device auto|cpu|cuda] [--work-dir WORK_DIR]
~~~

Короткие параметры: `-i`, `-o`, `-c`. `input` и `config-path` обязательны. `output` может
создать или переопределить `write_json_predictions_step` только в последовательном
конвейере; параллельный конвейер должен объявить шаг в YAML.

## sweep

~~~text
chimera-ml sweep --base-config BASE_CONFIG --sweep-config SWEEP_CONFIG [--sweep-name NAME] [--max-trials N] [--dry-run]
~~~

Короткие параметры: `-b`, `-s`, `-n`. `max-trials` должен быть не меньше 1. Подробности
см. в разделе [Переборы](../user-guide/sweeps.md).

## registry list

~~~text
chimera-ml registry list [--type TYPE]
~~~

`TYPE`: `datamodules`, `models`, `losses`, `metrics`, `optimizers`, `schedulers`, `callbacks`,
`collates`, `loggers` или `inference_steps`.

## plugins list

~~~text
chimera-ml plugins list [--group GROUP]
~~~

Группа по умолчанию — `chimera_ml.plugins`. Вложенные команды имеют форму
`command subcommand`.
