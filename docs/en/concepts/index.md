# Concepts

Chimera ML has a deliberately small runtime model:

```text
YAML config → registry factories → BuildContext → Trainer / pipeline
                                  ↘ callbacks and loggers
```

- [Architecture](architecture.md) explains the boundaries between the core
  package and task packages.
- [Configuration](configuration.md) explains the experiment YAML shape.
- [Registries](registries.md) explains name-based construction.
- [BuildContext](build-context.md) explains runtime metadata exchange.
- [Plugins](plugins.md) explains Python entry-point discovery.
