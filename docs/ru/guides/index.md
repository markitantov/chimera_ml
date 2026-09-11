# Руководства

Chimera ML отделяет инфраструктуру времени выполнения от компонентов
эксперимента, специфичных для задачи. Обычный рабочий процесс выглядит так:

1. установите `chimera-ml` и plugin package;
2. опишите данные, модель, функцию потерь, оптимизатор, метрики, логирование и
   колбэки в YAML;
3. проверьте конфигурацию командой `chimera-ml validate-config`;
4. запустите обучение, оценку, инференс или parameter sweep через CLI.

## Плагины и реестры

Компоненты регистрируются под именами `models`, `losses`, `metrics`, `callbacks`
и `inference_steps`. Плагины обнаруживаются через Python entry-point group
`chimera_ml.plugins`. Используйте эти команды, чтобы проверить текущий процесс:

```bash
chimera-ml plugins list
chimera-ml registry list --type models
chimera-ml registry list --type inference_steps
```

## Конфигурация и примеры

В корневом [README](https://github.com/markitantov/chimera_ml/blob/main/README.md)
приведены полная модель конфигурации и справочник команд CLI. README примеров
описывают специфичные для задач пути и зависимости:

- [оценка VA](https://github.com/markitantov/chimera_ml/blob/main/examples/va_estimation/README.md)
- [ORAGEN](https://github.com/markitantov/chimera_ml/blob/main/examples/oragen/README.md)
- [распознавание аффективных состояний](https://github.com/markitantov/chimera_ml/blob/main/examples/affective_states_recognition/README.md)

Эти примеры намеренно хранятся рядом с кодом и конфигурациями. Документация
проекта описывает общие понятия и навигацию, а не копирует каждый README
примера.
