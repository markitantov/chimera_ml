import torch

from chimera_ml.data.masking_collate import MaskingCollate


def test_masking_collate_pads_variable_lengths_and_builds_masks():
    collate = MaskingCollate(pad_sequences=True, include_legacy_meta_masks=True)
    batch = [
        {
            "inputs": {"audio": torch.randn(3, 4)},
            "target": torch.randn(2, 2),
            "meta": {},
        },
        {
            "inputs": {"audio": torch.randn(5, 4), "video": torch.randn(4, 8)},
            "target": torch.randn(4, 2),
            "meta": {},
        },
    ]

    out = collate(batch)

    assert out.inputs["audio"].shape == (2, 5, 4)
    assert out.inputs["video"].shape == (2, 4, 8)
    assert out.targets is not None
    assert out.targets.shape == (2, 4, 2)

    masks = out.get_masks()
    assert masks is not None
    assert torch.equal(masks["audio_mask"], torch.tensor([1.0, 1.0]))
    assert torch.equal(masks["video_mask"], torch.tensor([0.0, 1.0]))
    assert masks["sequence_mask"].shape == (2, 5)
    assert out.meta is not None
    assert "masks" in out.meta
    assert "sample_meta" in out.meta


def test_masking_collate_uses_explicit_sequence_masks_from_meta():
    collate = MaskingCollate(pad_sequences=True)
    batch = [
        {
            "inputs": {"audio": torch.randn(2, 3)},
            "target": torch.randn(2, 1),
            "meta": {"masks": {"sequence_mask": torch.tensor([True, False])}},
        },
        {
            "inputs": {"audio": torch.randn(2, 3)},
            "target": torch.randn(2, 1),
            "meta": {},
        },
    ]

    out = collate(batch)
    masks = out.get_masks()
    assert masks is not None
    assert torch.equal(masks["sequence_mask"][0], torch.tensor([True, False]))
