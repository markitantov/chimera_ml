import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.data.masking_collate import MaskingCollate
from chimera_ml.models._utils import _attention, _features, _mask, _mean, _sequence


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


def test_mask_rejects_incompatible_batch_dimensions():
    x = torch.ones(2, 3, 4)
    batch = Batch(inputs={}, targets=None, masks={"audio_mask": torch.ones(3)})

    with pytest.raises(ValueError, match="batch dimension"):
        _mask(batch, "audio", x)


def test_mask_applies_temporal_modality_masks_without_broadcasting_shared_masks():
    x = torch.ones(2, 3, 4)
    batch = Batch(
        inputs={},
        targets=None,
        masks={
            "audio_mask": torch.tensor([[1, 0, 1], [0, 1, 1]]),
            "sequence_mask": torch.tensor([[1, 1], [1, 0]]),
        },
    )

    result = _mask(batch, "audio", x)

    assert torch.equal(result, batch.masks["audio_mask"].bool())


def test_mask_rejects_invalid_sequence_mask_rank():
    x = torch.ones(2, 3, 4)
    batch = Batch(inputs={}, targets=None, masks={"sequence_mask": torch.ones(2, 3, 1)})

    with pytest.raises(ValueError, match="sequence_mask"):
        _mask(batch, "audio", x)


def test_mean_ignores_masked_values_and_returns_zero_for_empty_rows():
    values = torch.tensor([[[1.0, 2.0], [100.0, 200.0]], [[7.0, 8.0], [9.0, 10.0]]])
    valid = torch.tensor([[1, 0], [0, 0]])

    result = _mean(values, valid)

    assert torch.equal(result, torch.tensor([[1.0, 2.0], [0.0, 0.0]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_attention_is_finite_and_zero_for_fully_masked_keys(dtype: torch.dtype):
    query = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
    key = torch.randn(2, 2, 4, dtype=dtype, requires_grad=True)
    valid = torch.tensor([[1, 0], [0, 0]])

    result = _attention(query, key, valid)

    assert torch.isfinite(result).all()
    assert torch.equal(result[1], torch.zeros_like(result[1]))
    result.float().sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()


def test_attention_normalizes_only_over_available_keys():
    query = torch.tensor([[[1.0, 0.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    result = _attention(query, key, torch.tensor([[1, 0]]))

    assert torch.allclose(result, key[:, :1])


def test_sequence_and_features_reject_unsupported_encoder_outputs():
    with pytest.raises(ValueError, match="Expected \\[B, D\\] or \\[B, T, D\\]"):
        _sequence(torch.ones(2, 3, 4, 5))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_dim = 4
            self.encoders = torch.nn.ModuleDict({"audio": torch.nn.Flatten()})
            self.proj = torch.nn.ModuleDict({"audio": torch.nn.Identity()})

    batch = Batch(inputs={"audio": torch.ones(2, 3, 4)}, targets=None)
    with pytest.raises(ValueError, match="Feature 'audio' has width"):
        _features(Model(), batch, ("audio",))
