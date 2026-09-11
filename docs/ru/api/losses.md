# API losses

Losses получают \`ModelOutput\` и labeled \`Batch\`, возвращая scalar (или
elementwise, если поддерживается) tensor для optimization. Loss выбирается по
\`LOSSES\` registry key в YAML; direct construction удобен в custom loops и
тестах. Выбор objective: [Losses](../user-guide/losses.md).

## Contract

::: chimera_ml.losses.base.BaseLoss

## Regression

\`MSELoss\` и \`MAELoss\` используют PyTorch reductions и требуют matching
shapes у predictions и targets.

::: chimera_ml.losses.regression.MSELoss

::: chimera_ml.losses.regression.mse_loss

::: chimera_ml.losses.regression.MAELoss

::: chimera_ml.losses.regression.mae_loss

## Classification

Cross entropy и focal loss получают class-index targets. Focal loss дополнительно
поддерживает focusing, alpha weighting, reduction и label smoothing.

::: chimera_ml.losses.classification.CrossEntropyLoss

::: chimera_ml.losses.classification.cross_entropy_loss

::: chimera_ml.losses.focal.FocalLoss

::: chimera_ml.losses.focal.focal_loss

## Multilabel и agreement objectives

\`BCEWithLogitsLoss\` ожидает multi-hot targets. \`CCCLoss\` оптимизирует one
minus Lin's concordance correlation coefficient по flattened output dimensions.

::: chimera_ml.losses.multilabel.BCEWithLogitsLoss

::: chimera_ml.losses.multilabel.bce_with_logits_loss

::: chimera_ml.losses.ccc.CCCLoss

::: chimera_ml.losses.ccc.ccc_loss
