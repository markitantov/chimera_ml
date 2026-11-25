# Журналирование и MLflow

Полный реестр журналировщиков и значения по умолчанию приведены в разделе [Журналирование](../components/logging.md); здесь описан сценарий журналирования обучения.

## Локальный журнал в консоль и файл

```yaml
logging:
  - name: console_file_logger
    params:
      log_path: logs
      log_file: train.log
      console_level: INFO
      file_level: INFO
      file_mode: a
      encoding: utf-8
```

Журналировщик создаёт `logs/<experiment_name>/<run_name>/train.log`; CLI внедряет имя эксперимента и сгенерированное имя запуска. Журналы эпох содержат функцию потерь, число образцов и метрики. `train.params.log_every_steps` задаёт интервал записи пошаговых метрик в MLflow; колбэки получают `train_step/loss` на каждом обучающем батче.

## Журнал MLflow

```yaml
logging:
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
      experiment_name: va_estimation_fusion
      run_name: fusion
```

Журналировщик создаёт запуск MLflow, записывает параметры и метрики и принимает артефакты от колбэков. `tracking_uri` может указывать на локальную SQLite-базу или доступный сервис отслеживания. MLflow импортируется при создании журналировщика; при отсутствии зависимости установите её или удалите журналировщик. Локальный файл и MLflow могут работать вместе.

Метрики имеют ключи `train/loss`, `val/loss`, `val/mae`; пошаговая функция потерь — `train_step/loss`. Артефакты используют `snapshots`/`configs`, `predictions/<split>`, `figures/<split>`.

Для локального интерфейса используйте необязательную команду:

```bash
mlflow ui --backend-store-uri sqlite:///logs/mlflow.db
```

Один только URI отслеживания не запускает веб-интерфейс и удалённый сервер.
