# Каталог колбэков

Колбэки реагируют на события жизненного цикла обучения: запуск и завершение обучения, эпохи и пакеты. Они позволяют сохранять состояние, собирать результаты, останавливать обучение и отправлять уведомления, не смешивая эти действия с кодом Trainer.

Колбэк обучения наследует `BaseCallback` или реализует те же хуки. Trainer вызывает их в следующем порядке:

```text
on_fit_start
  для каждой эпохи: on_epoch_start → on_batch_end* → on_epoch_end
on_fit_end
```

`on_batch_end` получает `global_step` и журналы пакета; `on_epoch_end` — агрегированные журналы с префиксом выборки. Колбэки создаются в порядке списка YAML. Это колбэки жизненного цикла Trainer, а не инференсный `on_prediction(model, batch, output)`, который используется в `xai_step`.

| Ключ реестра | Хук | Назначение |
| --- | --- | --- |
| `checkpoint_callback` | `on_epoch_end` | Последняя и отобранные top-k контрольные точки |
| `early_stopping_callback` | `on_epoch_end` | `trainer.stop_training` после отсутствия улучшения |
| `snapshot_callback` | `on_fit_start` | Снимок исходного кода и конфигурации |
| `collect_predictions_callback` | start/end | CSV кэшированных предсказаний в MLflow |
| `plot_confusion_matrix_callback` | start/end | Графики матрицы ошибок классификации |
| `sweep_target_callback` | start/end | Лучшее скалярное значение цели перебора |
| `telegram_notifier_callback` | start/end | Итоговое уведомление Telegram |

Проверяйте реестр командой `chimera-ml registry list --type callbacks`.


## Пример матрицы ошибок

Колбэк предназначен только для классификации. Он включает сбор кэша и требует журналировщика MLflow, а также необязательной зависимости `matplotlib`.

~~~yaml
- name: plot_confusion_matrix_callback
  params:
    splits: [val]
    class_names: [negative, neutral, positive]
    artifact_path: figures
    filename_template: confusion_matrix_epoch_{epoch}.pdf
    title_template: "{split} Confusion Matrix (epoch {epoch})"
~~~

В `on_fit_start` колбэк включает сбор результатов, а в `on_epoch_end` записывает график для каждой выбранной выборки. Он не умеет интерпретировать предсказания регрессии.
