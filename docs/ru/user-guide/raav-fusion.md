# RAAV fusion

`raav_fusion_model` агрегирует temporal frame modalities с обучаемыми весами и
уточняет fused stream через latent queries над optional context modality.
Промежуточные tensors доступны в `ModelOutput.aux`.
