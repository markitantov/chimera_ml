# Гиперпараметрические переборы

Перебор принимает корректную конфигурацию обучения и запускает обычное обучение для каждого испытания. Сначала проверьте базовую конфигурацию: перебор не исправляет датасеты, ключи плагинов или отсутствующие мониторы.

## Сценарий полного перебора

`sweep.yaml` содержит либо `parameters`, либо `trials`, но не оба:

```yaml
parameters:
  optimizer.params.lr: [0.001, 0.0001]
  train.params.epochs: [3, 5]
```

Это даёт четыре комбинации полного перебора:

```bash
chimera-ml validate-config --config-path config.yaml
chimera-ml sweep --base-config config.yaml --sweep-config sweep.yaml --sweep-name lr-search --max-trials 4 --dry-run
chimera-ml sweep --base-config config.yaml --sweep-config sweep.yaml --sweep-name lr-search
```

Составные пути могут адресовать отображения и именованные элементы списков: `optimizer.params.lr`, `callbacks.checkpoint_callback.params.monitor`. `--max-trials` ограничивает число испытаний, `--dry-run` не запускает обучение.

## Сценарий Optuna

```yaml
method: optuna
n_trials: 10
study_name: lr-study
target:
  monitor: val/mae
  mode: min
parameters:
  optimizer.params.lr:
    type: float
    low: 0.00001
    high: 0.001
    log: true
  train.params.epochs:
    type: int
    low: 3
    high: 20
    step: 1
  model.params.variant:
    type: categorical
    choices: [small, large]
```

Тип `float` требует `low`/`high` и использует `step` или `log`, но не оба параметра. Тип `int` имеет `step` по умолчанию `1`; логарифмическая шкала для `int` требует `step: 1`. Тип `categorical` требует непустого `choices`. `target.mode` — `min`/`max`, а также псевдонимы `minimize`/`maximize`; цель по умолчанию — `val/loss`, `min`.

```bash
chimera-ml sweep -b config.yaml -s optuna.yaml --dry-run
chimera-ml sweep -b config.yaml -s optuna.yaml --max-trials 3
```

CLI добавляет `sweep_target_callback`, запускает обычный цикл обучения и записывает лучшее значение и эпоху.

## Артефакты

```text
<log_path>/<experiment_name>/_sweeps/<sweep-id>/
├── base_config.yaml
├── sweep_config.yaml
├── manifest.yaml
└── trial_configs/<label>-<short-id>-001.yaml
```

`manifest.yaml` содержит статус и записи испытаний; Optuna дополнительно сохраняет исследование, целевую функцию и лучшее испытание. Каждое испытание получает собственные артефакты запуска.
