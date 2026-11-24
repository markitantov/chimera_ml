# Data and loaders

A datamodule factory returns train, validation, and test DataLoaders. The
dataset supplies feature tensors and optional targets, masks, and metadata.

## Collates and masks

CHIMERA-ML provides two distinct built-in collates.

### MaskingCollate

MaskingCollate, registered as masking_collate, keeps the legacy behavior for
existing pipelines. It pads or stacks tensor fields, emits sequence_mask and
per-modality presence masks such as audio_mask, and can copy masks to legacy
meta["masks"] when configured. It remains available and is not replaced by the
new collate.

### MultimodalSequenceCollate

MultimodalSequenceCollate, registered as multimodal_sequence_collate, is for
feature modalities whose temporal lengths are independent.

- A sample vector [D] is collated as [B, D] with a [B, 1] temporal mask.
- A sequence [T, D] is padded per modality as [B, T_max, D].
- Each modality gets a float32 presence mask `<modality>_mask` with shape [B].
- Each temporal modality gets a boolean `<modality>_temporal_mask` with shape
  [B, T_max] and has boolean dtype.
- Missing rows are zero-filled. Padding uses pad_value.
- Explicit one-dimensional temporal masks may be supplied as
  `meta[<modality>_temporal_mask]`, `meta["masks"][<modality>_temporal_mask]`, or
  sample-level `sample["masks"][<modality>_temporal_mask]`; they are validated and combined
  with padding.
- pad_sequences, pad_value, and include_legacy_meta_masks are configurable.
- Zero-length sequences are supported.
- Timelines are not automatically aligned between modalities.

For example, audio sequences with lengths 2 and 1 become audio [B, 2, D_audio],
while video sequences with lengths 3 and 2 become video [B, 3, D_video]. The
two modalities keep their own temporal widths and masks; they are not padded to
one shared maximum.

Minimal registry configuration:

~~~yaml
collate:
  name: multimodal_sequence_collate
  params:
    pad_sequences: true
~~~

Fusion utilities consume these masks through _mask(). The dedicated temporal
mask is preferred and its shape is validated against the normalized [B, T, D]
features. Older [B] or [B, T] modality masks and compatible sequence_mask
formats remain supported.

## Multiple loaders

A loader may be a single DataLoader, a mapping, or a list/tuple. Training
supports single, round_robin, and weighted modes. Split names are normalized
for stable metric keys.

## Batch contract

~~~python
Batch(
    inputs={"audio": audio_tensor, "video": video_tensor},
    targets=target_tensor,
    masks={"audio_temporal_mask": audio_temporal_mask},
    meta={"id": sample_id},
)
~~~

Batch.get_masks accepts no argument for all masks or a specific flat key. For
compatibility it also reads legacy nested masks from masks["mask"] and
meta["masks"].
