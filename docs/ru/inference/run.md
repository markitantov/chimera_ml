# Запуск инференса

Нужны Python 3.12, установленный CHIMERA-ML, зарегистрированный шаг предсказания, существующий входной файл и совместимые веса. Инференс не обучает модель и не вычисляет метрики по разметке.

## 1. Установка среды и обязательных компонентов

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# Если шаг предсказания находится в плагине:
python -m pip install -e path/to/your_plugin
chimera-ml --help
chimera-ml doctor
chimera-ml plugins list
chimera-ml registry list --type inference_steps
```

Плагин нужно устанавливать в то же окружение, из которого запускается CLI. Ключ, видимый в другом Python-окружении, недоступен текущему процессу. Команда `doctor` показывает базовую диагностику, а `registry list` подтверждает реальные ключи.

## 2. Подготовка входа и контрольной точки

CLI проверяет, что `--input` существует:

```bash
test -f data/example.bin
```

Подготовьте локальную контрольную точку или URL HTTP(S), совместимый с шагом предсказания. `resolve_checkpoints_step` только разрешает путь и скачивает файл в кэш; он не строит модель и не загружает `state_dict`. Это делает последующий шаг предсказания.

## 3. Создание YAML инференса

Ниже `your_prediction_step` — намеренно иллюстративный ключ. Замените его на зарегистрированный ключ и реальные параметры плагина; ядро не знает универсального формата входа или весов.

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

Проверяйте ключи командой `chimera-ml registry list --type inference_steps`. Небольшой полностью исполняемый синтетический конвейер с регистрацией шага приведён в [пользовательских шагах предсказания](custom-steps.md).

## 4. Проверка правильной конфигурации

`chimera-ml validate-config` предназначена для `ExperimentConfig` обучения и оценки и не является валидатором YAML инференса. Для инференса проверьте поля по [схеме конвейера](pipeline.md), реестр шагов и затем дайте CLI построить `InferenceConfig`:

```bash
chimera-ml inference   --input data/example.bin   --config-path configs/inference.yaml   --output outputs/example.json   --device auto   --work-dir outputs/work
```

`--input/-i` и `--config-path/-c` обязательны. `--device` принимает только `auto`, `cpu` или `cuda`; `auto` выбирает CUDA при доступности, иначе CPU. `--output/-o` в последовательном конвейере создаёт средство записи, если его нет, или переопределяет его `output_path`. В параллельном конвейере средство записи должно быть явно указано в YAML с зависимостями. `--work-dir` задаёт рабочий каталог; без него CLI создаёт временный `chimera-inference-*` каталог.

Стандартное средство записи создаёт только такую полезную нагрузку:

```json
{
  "input": "data/example.bin",
  "predictions": [{"label": "example", "score": 0.8}]
}
```

Формат `predictions` определяется плагином. Ядро не добавляет метки, калибровку или метаданные задачи. `print_json_predictions_step` печатает ту же полезную нагрузку без файла. Если `--output` не задан, средство записи последовательного конвейера меняет суффикс входного файла на `.json`.

## 5. Запуск и проверка результата

1. Установите ядро и плагин, проверьте регистрацию.
2. Подготовьте входной файл и контрольную точку.
3. Создайте YAML с реальными шагами предсказания и вывода.
4. Проверьте поля по схеме конвейера; не используйте для этой цели `validate-config`.
5. Запустите CLI и проверьте JSON.
6. Для кэша и промежуточных файлов используйте постоянный `--work-dir`; без него каталог будет временным. Журналы и прикладные артефакты создаёт сам шаг плагина.

В отличие от `chimera-ml eval`, инференс не требует обучающей или валидационной выборки и не вычисляет метрики автоматически.
