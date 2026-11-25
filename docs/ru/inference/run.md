# Запуск инференса

Нужны Python 3.12, установленный CHIMERA-ML, зарегистрированный prediction step, существующий входной файл и совместимые веса. Инференс не обучает модель и не считает метрики по разметке.

## Установка и проверка окружения

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# Если prediction step находится в плагине:
python -m pip install -e path/to/your_plugin
chimera-ml --help
chimera-ml doctor
chimera-ml plugins list
chimera-ml registry list --type inference_steps
```

Плагин нужно устанавливать в то же окружение, из которого запускается CLI. Ключ, видимый в другом Python-окружении, недоступен текущему процессу. Команда `doctor` показывает базовую диагностику, а `registry list` подтверждает реальные ключи.

## Подготовка входа и checkpoint

CLI проверяет, что `--input` существует:

```bash
test -f data/example.bin
```

Подготовьте локальный checkpoint или URL HTTP(S), совместимый с prediction step. `resolve_checkpoints_step` только разрешает путь и скачивает файл в кэш; он не строит модель и не загружает `state_dict`. Это делает downstream prediction step.

## Конфигурация

Ниже `your_prediction_step` — намеренно иллюстративный ключ. Замените его на зарегистрированный ключ и реальные параметры плагина; core не знает универсального формата входа или весов.

```yaml
pipeline:
  name: application-demo
  parallel: false
runtime:
  device: auto
steps:
  - name: resolve_checkpoints_step
    params:
      cache_dir: model_cache
      checkpoints:
        fusion: /absolute/path/to/fusion.pt
  - name: your_prediction_step
    params:
      checkpoint_key: fusion
  - name: write_json_predictions_step
    params:
      output_path: outputs/predictions.json
```

Проверяйте ключи командой `chimera-ml registry list --type inference_steps`. Маленький полностью исполняемый синтетический pipeline с регистрацией шага приведён в [пользовательских prediction steps](custom-steps.md).

## Проверка и запуск

`chimera-ml validate-config` предназначена для `ExperimentConfig` обучения/оценки и не является валидатором inference YAML. Для inference проверьте поля по [схеме pipeline](pipeline.md), реестр шагов и затем дайте CLI построить `InferenceConfig`:

```bash
chimera-ml inference   --input data/example.bin   --config-path configs/inference.yaml   --output outputs/example.json   --device auto   --work-dir outputs/work
```

`--input/-i` и `--config-path/-c` обязательны. `--device` принимает только `auto`, `cpu` или `cuda`; `auto` выбирает CUDA при доступности, иначе CPU. `--output/-o` в последовательном pipeline создаёт writer, если его нет, или переопределяет его `output_path`. В parallel pipeline writer должен быть явно записан в YAML с зависимостями. `--work-dir` задаёт рабочий каталог; без него CLI создаёт временный `chimera-inference-*` каталог.

Стандартный writer создаёт только такой payload:

```json
{
  "input": "data/example.bin",
  "predictions": [{"label": "example", "score": 0.8}]
}
```

Форма `predictions` принадлежит плагину. Core не добавляет labels, calibration или task metadata. `print_json_predictions_step` печатает тот же payload без файла. Если `--output` не задан, последовательный writer меняет суффикс входа на `.json`.

## Рабочие каталоги и порядок действий

1. Установите core и плагин, проверьте регистрацию.
2. Подготовьте вход и checkpoint.
3. Создайте YAML с реальными prediction/output steps.
4. Проверьте поля по схеме pipeline; `validate-config` для этой цели не используйте.
5. Запустите CLI и проверьте JSON.
6. Для кэша и промежуточных файлов используйте постоянный `--work-dir`; без него каталог временный. Журналы и прикладные артефакты создаёт сам plugin step.

В отличие от `chimera-ml eval`, inference не требует train/validation split и не считает метрики автоматически.
