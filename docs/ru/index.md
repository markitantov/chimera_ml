# Chimera ML

CHIMERA-ML — конфигурационный PyTorch framework для построения, обучения, оценки и запуска uni-modal и multi-modal экспериментов. Он предоставляет YAML-конфигурацию, registries, Trainer, callbacks, loggers, checkpoints и inference pipelines; task-specific datasets и модели могут находиться в plugins.

## С чего начать

- [Начало работы](getting-started/index.md): установка, проверка окружения и первый конфигурационный check.
- [Обучение](training/index.md): данные, конфигурация, обучение, evaluation и artifacts.
- [Inference](user-guide/inference.md): запуск sequential или DAG pipeline по входному файлу.
- [Компоненты](components/index.md): контракты Batch/ModelOutput, fusion-модели и остальные built-in components.
- [Примеры](tutorials/index.md): VA, ORAGEN и affective-states проекты репозитория.
- [Расширение](extending/index.md): регистрация собственных datasets, моделей, losses, metrics, callbacks и inference steps.
- [Справочник](api/index.md): Python API и точные CLI/configuration references.

## Основная схема

Обучение: `Dataset -> DataModule/DataLoader -> Batch -> Model -> ModelOutput -> loss/metrics -> Trainer -> checkpoints`. Inference отделён: `input -> InferencePipeline -> registered steps -> artifacts/predictions`. YAML выбирает компоненты по registry key, а `params` передаются их factory.

Core не содержит датасеты задач и не подставляет пути автоматически. Установите plugin, нужный для примера или конфигурации, измените пути к ресурсам и сначала запустите `validate-config`.

~~~bash
pip install chimera-ml
chimera-ml doctor
chimera-ml plugins list
chimera-ml registry list
~~~
