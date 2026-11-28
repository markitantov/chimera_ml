# Установка

## Требования и варианты установки

CHIMERA-ML поддерживает Python `>=3.12`. Ядро объявляет зависимости PyTorch `>=2.2,<3.0`, NumPy, PyYAML, Typer, tqdm, matplotlib, MLflow, Optuna и requests. CUDA-сборка и драйвер PyTorch выбираются под конкретную машину; CHIMERA-ML не устанавливает драйвер CUDA.

Установка опубликованного пакета в виртуальное окружение:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install chimera-ml
```

Доступны две равнозначные точки входа:

```bash
chimera-ml --help
chimera_ml --help
```

Для разработки репозитория:

```bash
poetry install --with dev,docs
poetry run chimera-ml doctor
```

Группа `docs` нужна только для сборки документации. У плагинов из примеров есть собственные зависимости; например:

```bash
python -m pip install -e examples/va_estimation
```

Это устанавливает точку входа `chimera_ml.plugins` и делает фабрики плагина доступными при запуске процесса. Датасет и файлы признаков не скачиваются; пути к ним нужно подготовить отдельно.

## Проверка окружения

```bash
chimera-ml doctor
chimera-ml plugins list
chimera-ml registry list
```

`doctor` показывает версии Python и PyTorch, доступность CUDA, наличие MLflow, количество реестров и плагинов. `plugins list` показывает точки входа, а `registry list` — реально загруженные ключи.

## Воспроизводимость зависимостей

Диапазоны версий в метаданных пакета описывают поддерживаемые установки. `poetry.lock` фиксирует воспроизводимое окружение разработки и используется CI в locked-режиме. Для научного эксперимента следует сохранять собственную фиксацию окружения рядом с чекпоинтами; lock разработки не заменяет снимок эксперимента. CI также отдельно проверяет наборы lowest-direct и latest, не зависящие от lock-файла.

## CPU и CUDA

`train.params.device` обычно принимает `cuda` или `cpu`. Trainer переходит на CPU, если CUDA недоступна. `mixed_precision: true` включает AMP и масштабирование градиентов только на CUDA; на CPU сохраняется полная точность.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Для GPU нужны совместимые CUDA-сборка PyTorch и драйвер. Режим CPU подходит для проверки конфигурации и регистрации плагинов.

## Основные точки входа CLI

```text
chimera-ml validate-config --config-path CONFIG.yaml
chimera-ml doctor
chimera-ml train --config-path CONFIG.yaml
chimera-ml eval --config-path CONFIG.yaml --checkpoint-path CHECKPOINT.pt [--with-features]
chimera-ml inference --input INPUT --config-path INFERENCE.yaml [--output OUTPUT.json] [--device auto|cpu|cuda] [--work-dir DIR]
chimera-ml sweep --base-config CONFIG.yaml --sweep-config SWEEP.yaml [--dry-run]
chimera-ml registry list [--type TYPE]
chimera-ml plugins list [--group chimera_ml.plugins]
```

Для отслеживания MLflow нужны MLflow и локальный или доступный URI отслеживания; Optuna нужен для `method: optuna`; колбэк Telegram дополнительно требует `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID`.
