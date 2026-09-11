# CLI workflows

Точный syntax находится в [CLI reference](../cli/reference.md).

~~~bash
chimera-ml validate-config -c config.yaml
chimera-ml train -c config.yaml
chimera-ml eval -c test.yaml --checkpoint-path path/to/last.pt
chimera-ml inference -i input.mp4 -o prediction.json -c inference.yaml
chimera-ml sweep -b config.yaml -s sweep.yaml --dry-run
~~~

Используйте doctor для environment problems, plugins list для entry-point
discovery и registry list для registration. Доступен alias chimera_ml.
