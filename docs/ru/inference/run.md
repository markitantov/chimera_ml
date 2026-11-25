# Запуск инференса

Нужны Python 3.12, установленный CHIMERA-ML, зарегистрированный шаг предсказания, входной файл и совместимые веса. Инференс не обучает модель и не проверяет качество по разметке.

## Установка и проверка

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
chimera-ml --help
chimera-ml plugins list
chimera-ml registry list --type inference_steps
```

Плагин устанавливайте отдельной командой `python -m pip install -e path/to/plugin` (или обычным `pip install` для опубликованного пакета). Входные и выходные каталоги должны быть доступны процессу.

## Команда

```bash
chimera-ml inference   --input data/example.bin   --config-path configs/inference.yaml   --output outputs/example.json   --device auto   --work-dir outputs/work
```

`--input` и `--config-path` обязательны. `--device` принимает `auto`, `cpu` или `cuda`; `auto` выбирает CUDA только если она доступна. `--output` создаёт или переопределяет встроенный JSON writer в последовательном pipeline. Для parallel pipeline writer нужно описать явно. `--work-dir` задаёт рабочий каталог; без него CLI создаёт временный каталог.

После выполнения writer создаёт JSON с полями `input` и `predictions`. Если output не задан, последовательный writer использует имя входа с заменённым суффиксом `.json`. Пути чекпоинтов и журналы зависят от прикладного шага; ищите их в указанном `work-dir` или в путях из конфигурации.

## Полный порядок

1. Установите core и плагин, затем проверьте его в `registry list`.
2. Подготовьте вход и доступный локальный файл весов либо URL HTTP(S).
3. Создайте YAML с зарегистрированным prediction step и writer.
4. Проверьте структуру командой `chimera-ml validate-config --config-path configs/inference.yaml` (параметры, не относящиеся к этой версии CLI, проверяйте через `chimera-ml validate-config --help`).
5. Запустите команду выше.
6. Проверьте JSON и артефакты в work directory.

В отличие от `eval`, inference не требует train/validation split и не считает метрики сам по себе. Совместимость checkpoint, формат входа и JSON-сериализация — ответственность prediction step.
