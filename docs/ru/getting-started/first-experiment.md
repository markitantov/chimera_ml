# Первый эксперимент

В пошаговом руководстве используется плагин оценки VA из репозитория. Это не демонстрация без данных: нужны зависимости плагина, данные признаков и меток VA, доступный для записи каталог и пути из YAML.

## 1. Установить среду выполнения и плагин

```bash
python -m pip install chimera-ml
python -m pip install -e examples/va_estimation
chimera-ml doctor
chimera-ml plugins list
```

Плагин регистрирует `fusion_va_datamodule`, `fusion_model`, `fusion_ccc_mse_loss`, `va_ccc_metric` и `fusion_framewise_callback`. Проверяйте ключи своей среды через `chimera-ml registry list`.

## 2. Подготовить данные и пути

Измените `examples/va_estimation/configs/multimodal_train.yaml`: пути к признакам и аннотациям находятся в `data.params`. Плагин ожидает файлы признаков VA и метки; фреймворк их не создаёт. Если оставляете `telegram_notifier_callback`, задайте переменные или удалите колбэк:

```bash
export TELEGRAM_BOT_TOKEN='your-token'
export TELEGRAM_CHAT_ID='your-chat-id'
```

Не помещайте учётные данные в YAML или git.

## 3. Проверить конфигурацию и запустить обучение

```bash
chimera-ml validate-config   --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml train   --config-path examples/va_estimation/configs/multimodal_train.yaml
```

Проверка имеет структурный характер. Обучение дополнительно требует ключи плагина, пути к датасету, формы признаков и существующий монитор колбэка. Пример просит CUDA и mixed precision; при недоступной CUDA Trainer перейдёт на CPU.

## 4. Найти журналы и контрольные точки

В примере `log_path: logs`, `experiment_name: va_estimation_fusion`. CLI передаёт сгенерированное имя запуска колбэкам:

```text
logs/
└── va_estimation_fusion/
    └── <generated-run-name>/
        ├── train.log
        ├── checkpoints/
        │   ├── last.pt
        │   └── epoch=...pt
        ├── <config-file>.yaml
        └── code.zip
```

Конфигурацию и `code.zip` создаёт колбэк снимка при включённых параметрах. Контрольная точка содержит состояния модели и оптимизатора, эпоху, глобальный шаг и состояние планировщика, если он задан. Журналировщик MLflow записывает метрики и артефакты в хранилище отслеживания.

## 5. Оценить контрольную точку

```bash
chimera-ml eval   --config-path examples/va_estimation/configs/multimodal_test.yaml   --checkpoint-path logs/va_estimation_fusion/your_run/checkpoints/last.pt
```

Оценка заново строит DataModule, модель, функцию потерь, метрики и колбэки, загружает `model_state_dict` из полезной нагрузки или необёрнутого словаря состояния и требует точного совпадения архитектуры. Это не инференс развёртывания.

## Для своего проекта

Сохраните структуру YAML фреймворка, но замените специфичные для плагина `data`, `model`, `loss`, `metrics` и колбэки своими зарегистрированными компонентами. См. [Основные понятия](essential-concepts.md), [подготовку наборов данных](../user-guide/data.md) и [конфигурацию](../user-guide/configuration.md).
