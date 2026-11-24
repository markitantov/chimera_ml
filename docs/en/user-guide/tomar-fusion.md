# TOMAR fusion

`tomar_fusion_model` keeps an anchor modality as the base representation and
adds gated residual corrections from available extra modalities. It exposes
the anchor, residual gates, residuals, and normalized fused representation in
`ModelOutput.aux`.
