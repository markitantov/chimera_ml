# Models

For the structured component catalog, see [Models](../components/models.md) and the [fusion comparison](../components/fusion.md).

A model subclasses BaseModel and implements forward(batch: Batch) ->
ModelOutput. ModelOutput.preds is the task prediction tensor; ModelOutput.aux
contains optional fused representations, validity masks, weights, and attention
diagnostics.

## Built-in fusion models

The four built-in fusion models accept precomputed features directly from
Batch.inputs. They create modality projections and prediction layers internally;
no external encoder dictionary or prediction head is passed to the constructor.

| Registry key | Class | Prediction shape | Main mechanism |
| --- | --- | --- | --- |
| tfs_fusion_model | TFSFusionModel | [B, output_dim] | MASAI-style directed hierarchy and statistical pooling |
| tfs_dcmmmoe_fusion_model | TFSDCMMOEFusionModel | [B, output_dim] | TFS hierarchy with adaptive masked pair gates |
| raav_fusion_model | RAAVFusionModel | [B, T, output_dim] | Frame fusion and context-latent attention |
| tomar_fusion_model | TOMARFusionModel | [B, output_dim] | Anchor-conditioned gated residuals |

The common constructor fields are input_dims, shared_dim, and output_dim.
RAAV additionally requires frame_modalities and context_modality. TOMAR
requires anchor_modality. See the individual guides for masks, auxiliary
values, and edge cases:

- [TFS](tfs-fusion.md)
- [TFS-DCMMOE](tfs-dcmmmoe-fusion.md)
- [RAAV](raav-fusion.md)
- [TOMAR](tomar-fusion.md)

A modality may be absent from Batch.inputs or unavailable for a sample when
its mask is false. TFS and TFS-DCMMOE need at least two modality tensors in
Batch.inputs to form a pair. If all their temporal positions are masked, fused
is zero but the prediction layer still returns a prediction. RAAV requires
equal temporal length across present frame modalities; its context timeline may
differ. TOMAR requires the anchor for every sample.

Models can be instantiated as Python classes or built through the MODELS
registry from YAML. The [API reference](../api/models.md) documents the public
classes and factories.
