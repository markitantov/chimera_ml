# Models

Every model implements BaseModel.forward(Batch) -> ModelOutput. The built-in
fusion models consume feature tensors from Batch.inputs, create modality
projections internally, and return predictions in ModelOutput.preds.

| Registry key | Class | Output | Description |
| --- | --- | --- | --- |
| tfs_fusion_model | TFSFusionModel | [B, output_dim] | Hierarchical directed pair fusion |
| raav_fusion_model | RAAVFusionModel | [B, T, output_dim] | Frame fusion with contextual refinement |
| tomar_fusion_model | TOMARFusionModel | [B, output_dim] | Anchor-based residual fusion |

See the individual guides for constructor arguments, masks, auxiliary outputs,
and examples:

- TFS: tfs-fusion.md
- RAAV: raav-fusion.md
- TOMAR: tomar-fusion.md

Models can be built directly in Python or through the MODELS registry.
