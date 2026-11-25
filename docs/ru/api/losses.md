# API функций потерь

Функции потерь получают `ModelOutput` и размеченный `Batch`, возвращая скалярный
(или поэлементный, если поддерживается) тензор для оптимизации. Функция потерь
выбирается по ключу реестра `LOSSES` в YAML; прямое создание удобно в пользовательских
циклах и тестах. Выбор цели описан в разделе [Функции потерь](../user-guide/losses.md).

## Контракт

::: chimera_ml.losses.base.BaseLoss

## Регрессия

`MSELoss` и `MAELoss` используют редукции PyTorch и требуют совпадающих
форм предсказаний и целевых значений.

::: chimera_ml.losses.regression.MSELoss

::: chimera_ml.losses.regression.mse_loss

::: chimera_ml.losses.regression.MAELoss

::: chimera_ml.losses.regression.mae_loss

## Классификация

Cross entropy и focal loss получают целевые значения в виде индексов классов.
Focal loss дополнительно поддерживает фокусировку, взвешивание `alpha`, редукцию
и сглаживание меток.

::: chimera_ml.losses.classification.CrossEntropyLoss

::: chimera_ml.losses.classification.cross_entropy_loss

::: chimera_ml.losses.focal.FocalLoss

::: chimera_ml.losses.focal.focal_loss

## Мультиметочные и согласующие цели

\`BCEWithLogitsLoss\` ожидает многометочные целевые значения. \`CCCLoss\` минимизирует
единицу минус коэффициент конкордации Лина по сплющенным размерностям результата.

::: chimera_ml.losses.multilabel.BCEWithLogitsLoss

::: chimera_ml.losses.multilabel.bce_with_logits_loss

::: chimera_ml.losses.ccc.CCCLoss

::: chimera_ml.losses.ccc.ccc_loss
