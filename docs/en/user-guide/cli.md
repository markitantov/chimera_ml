# CLI workflows

The [CLI reference](../cli/reference.md) contains exact syntax.

~~~bash
chimera-ml validate-config -c config.yaml
chimera-ml train -c config.yaml
chimera-ml eval -c test.yaml --checkpoint-path path/to/last.pt
chimera-ml inference -i input.mp4 -o prediction.json -c inference.yaml
chimera-ml sweep -b config.yaml -s sweep.yaml --dry-run
~~~

Use doctor for environment problems. Use plugins list for entry-point
discovery and registry list for registration. The underscore executable
chimera_ml is also available.
