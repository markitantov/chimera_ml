# Устранение проблем

- **Шаг не найден.** Установите plugin в то же окружение и выполните `chimera-ml registry list --type inference_steps`. В YAML нужен registry key, а не имя Python-класса.
- **Неверная команда валидации.** `chimera-ml validate-config` проверяет `ExperimentConfig` обучения/оценки, не inference YAML. Для inference сверяйте поля с [Pipeline configuration](pipeline.md) и registry.
- **Нет `predictions`.** Prediction step должен вызвать `ctx.set_artifact("predictions", value)` до print/write шага.
- **Не найден вход.** CLI проверяет существование `--input`; подготовьте файл заранее.
- **CUDA недоступна.** Используйте `--device cpu` или `runtime.device: cpu`; `auto` выберет CPU, если CUDA недоступна.
- **Конфликт артефактов.** В parallel DAG свяжите перезаписывающий узел через `after` с владельцем либо используйте разные имена.
- **Cycle или неизвестный `after`.** Проверьте уникальность `id` и направление зависимостей.
- **Не скачивается checkpoint.** Разрешены только HTTP(S); проверьте сеть, права `work-dir` и место для `.part`.
- **XAI не запускается.** Модель и вложенные модули должны быть в `eval()`, а prediction step должен иметь атрибут `on_prediction` (он может быть `None` до wrapping).
- **Нет XAI в JSON.** `write_json_predictions_step` экспортирует только `input` и `predictions`; получите `result.get_artifact("xai")` и сохраните отдельным writer.
- **JSON не сериализуется.** Преобразуйте tensors и другие значения в JSON-compatible форму в прикладном шаге.
