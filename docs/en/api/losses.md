# Losses API

Losses consume a \`ModelOutput\` and a labeled \`Batch\`, returning a scalar (or
elementwise, when supported) tensor for optimization. Select a loss with its
\`LOSSES\` registry key in YAML; direct construction is useful for tests and
custom loops. See [Losses](../user-guide/losses.md) for choosing a task-appropriate
objective.

## Contract

::: chimera_ml.losses.base.BaseLoss

## Regression

\`MSELoss\` and \`MAELoss\` delegate to PyTorch reductions and expect predictions
and targets with matching shapes.

::: chimera_ml.losses.regression.MSELoss

::: chimera_ml.losses.regression.mse_loss

::: chimera_ml.losses.regression.MAELoss

::: chimera_ml.losses.regression.mae_loss

## Classification

Cross entropy and focal loss consume class-index targets. Focal loss additionally
supports focusing, alpha weighting, reduction, and label smoothing.

::: chimera_ml.losses.classification.CrossEntropyLoss

::: chimera_ml.losses.classification.cross_entropy_loss

::: chimera_ml.losses.focal.FocalLoss

::: chimera_ml.losses.focal.focal_loss

## Multilabel and agreement objectives

\`BCEWithLogitsLoss\` expects multi-hot targets. \`CCCLoss\` optimizes one minus
Lin's concordance correlation coefficient across flattened output dimensions.

::: chimera_ml.losses.multilabel.BCEWithLogitsLoss

::: chimera_ml.losses.multilabel.bce_with_logits_loss

::: chimera_ml.losses.ccc.CCCLoss

::: chimera_ml.losses.ccc.ccc_loss
