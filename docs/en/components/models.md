# Models

## Model contract

A model implements BaseModel.forward(batch: Batch) -> ModelOutput. It consumes
Batch.inputs and returns the main prediction in preds; optional architecture
diagnostics are stored in aux.

## Built-in fusion models

| Key | Class | Principle | Output |
| --- | --- | --- | --- |
| tfs_fusion_model | TFSFusionModel | Directed hierarchical pair fusion | [B, output_dim] |
| raav_fusion_model | RAAVFusionModel | Frame fusion with context corrections | [B,T,output_dim] |
| tomar_fusion_model | TOMARFusionModel | Anchor-conditioned residual corrections | [B, output_dim] |

All models accept feature tensors [B,D] or [B,T,D]; RAAV returns a sequence.
Missing inputs and masks are handled as described in each model guide. Auxiliary
values are diagnostics, not direct influence scores.
