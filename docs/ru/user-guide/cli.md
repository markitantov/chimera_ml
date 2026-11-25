# Сценарии CLI

Точный синтаксис приведён в [справочнике CLI](../cli/reference.md).

~~~bash
chimera-ml validate-config -c config.yaml
chimera-ml train -c config.yaml
chimera-ml eval -c test.yaml --checkpoint-path path/to/last.pt
chimera-ml inference -i input.mp4 -o prediction.json -c inference.yaml
chimera-ml sweep -b config.yaml -s sweep.yaml --dry-run
~~~

Используйте `doctor` для диагностики проблем окружения, `plugins list` — для
поиска плагинов через точки входа, а `registry list` — для проверки регистрации
компонентов. Также доступен псевдоним `chimera_ml`.
