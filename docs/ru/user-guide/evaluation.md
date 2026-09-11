# Evaluation

Оцените checkpoint без optimization:

~~~bash
chimera-ml eval \
  --config-path config.yaml \
  --checkpoint-path logs/experiment/run/checkpoints/last.pt
~~~

Команда строит те же data/model/loss/metrics/callback components, что и train.
Checkpoint загружается с weights_only=True на CPU и может быть mapping с
model_state_dict или raw state dictionary. load_state_dict(strict=True)
требует точного соответствия архитектуры.

Train, validation и test loaders datamodule объединяются в split names.
Evaluation выполняется как один epoch через Trainer.evaluate, запускает
callbacks, обновляет metrics и может cache predictions. --with-features
требует ModelOutput.aux["features"] либо прямой feature_extractor.
