# TOMAR fusion

`tomar_fusion_model` сохраняет anchor modality как базовое представление и
добавляет gated residual corrections от доступных дополнительных modalities.
Anchor, gates, residuals и fused representation доступны в `ModelOutput.aux`.
