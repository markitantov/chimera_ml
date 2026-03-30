import torch

from chimera_ml.core.batch import Batch


def test_get_masks_returns_batch_masks():
    batch = Batch(
        inputs={},
        targets=None,
        masks={
            "sequence_mask": torch.tensor([[True, False]]),
            "audio_mask": torch.tensor([1.0, 0.0]),
        },
    )

    got = batch.get_masks()
    assert got is not None
    assert torch.equal(got["audio_mask"], torch.tensor([1.0, 0.0]))


def test_get_masks_returns_specific_mask():
    batch = Batch(
        inputs={},
        targets=None,
        masks={
            "sequence_mask": torch.tensor([[True, False]]),
            "audio_mask": torch.tensor([1.0, 0.0]),
        },
    )

    assert torch.equal(batch.get_masks("audio_mask"), torch.tensor([1.0, 0.0]))
    assert batch.get_masks("missing") is None


def test_get_masks_falls_back_to_meta_masks():
    batch = Batch(
        inputs={},
        targets=None,
        meta={
            "masks": {"sequence_mask": torch.tensor([[True]]), "audio_mask": torch.tensor([1.0])}
        },
    )

    got = batch.get_masks()
    assert got is not None
    assert torch.equal(got["sequence_mask"], torch.tensor([[True]]))
