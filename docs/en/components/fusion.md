# Fusion comparison

The built-in fusion models share the BaseModel.forward contract but implement
different interaction schemes.

| Model | Required inputs | Output | Main diagnostics |
| --- | --- | --- | --- |
| TFS | At least two configured modalities | [B, output_dim] | Pair and group attention, validity, weights |
| RAAV | At least one frame modality and context list | [B,T,output_dim] | Frame weights, context attention, corrections |
| TOMAR | Anchor modality for every sample | [B, output_dim] | Anchor, residuals, sigmoid coefficients |

TFS creates ordered pair interactions and refines groups. RAAV computes a
weighted frame representation and adds one contextual correction per configured
context modality. TOMAR pools the anchor and each additional modality, then
adds a learned gated correction. The auxiliary tensors are retained for XAI
comparison but are not themselves sensitivity scores.
