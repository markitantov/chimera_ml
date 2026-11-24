import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.data.masking_collate import MaskingCollate
from chimera_ml.models._utils import _mask


def test_masking_collate_masks_are_combined_for_temporal_features():
    batch = MaskingCollate()(
        [
            {"inputs": {"text": torch.ones(2, 4)}},
            {"inputs": {}},
        ]
    )

    result = _mask(batch, "text", batch.inputs["text"])
    expected = batch.masks["sequence_mask"] & batch.masks["text_mask"].bool()[:, None]

    assert torch.equal(result, expected)
    assert torch.equal(result[0], torch.tensor([True, True]))
    assert torch.equal(result[1], torch.tensor([False, False]))


def test_mask_supports_canonical_presence_masks_and_rejects_bad_shapes():
    x = torch.ones(2, 3, 4)
    batch = Batch(
        inputs={},
        targets=None,
        masks={"audio_mask": torch.tensor([1, 0])},
    )

    result = _mask(batch, "audio", x)

    assert torch.equal(result, torch.tensor([[True, True, True], [False, False, False]]))

    bad = Batch(inputs={}, targets=None, masks={"audio_mask": torch.ones(2, 2, 2)})
    with pytest.raises(ValueError, match="Mask for 'audio'"):
        _mask(bad, "audio", x)
