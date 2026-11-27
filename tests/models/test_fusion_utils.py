import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.data.masking_collate import MaskingCollate
from chimera_ml.models.utils import attention, features, mask, masked_softmax, mean, sequence


def testmasking_collatemasks_are_combined_for_temporalfeatures():
    batch = MaskingCollate()(
        [
            {"inputs": {"text": torch.ones(2, 4)}},
            {"inputs": {}},
        ]
    )

    result = mask(batch, "text", batch.inputs["text"])
    expected = batch.masks["sequencemask"] & batch.masks["textmask"].bool()[:, None]

    assert torch.equal(result, expected)
    assert torch.equal(result[0], torch.tensor([True, True]))
    assert torch.equal(result[1], torch.tensor([False, False]))


def testmask_supports_canonical_presencemasks_and_rejects_bad_shapes():
    x = torch.ones(2, 3, 4)
    batch = Batch(
        inputs={},
        targets=None,
        masks={"audiomask": torch.tensor([1, 0])},
    )

    result = mask(batch, "audio", x)

    assert torch.equal(result, torch.tensor([[True, True, True], [False, False, False]]))

    bad = Batch(inputs={}, targets=None, masks={"audiomask": torch.ones(2, 2, 2)})
    with pytest.raises(ValueError, match="Mask for 'audio'"):
        mask(bad, "audio", x)


def testmask_rejects_incompatible_batch_dimensions():
    x = torch.ones(2, 3, 4)
    batch = Batch(inputs={}, targets=None, masks={"audiomask": torch.ones(3)})

    with pytest.raises(ValueError, match="batch dimension"):
        mask(batch, "audio", x)


def testmask_applies_temporal_modalitymasks_without_broadcasting_sharedmasks():
    x = torch.ones(2, 3, 4)
    batch = Batch(
        inputs={},
        targets=None,
        masks={
            "audiomask": torch.tensor([[1, 0, 1], [0, 1, 1]]),
            "sequencemask": torch.tensor([[1, 1], [1, 0]]),
        },
    )

    result = mask(batch, "audio", x)

    assert torch.equal(result, batch.masks["audiomask"].bool())


def testmask_rejects_invalidsequencemask_rank():
    x = torch.ones(2, 3, 4)
    batch = Batch(inputs={}, targets=None, masks={"sequencemask": torch.ones(2, 3, 1)})

    with pytest.raises(ValueError, match="sequencemask"):
        mask(batch, "audio", x)


def testmean_ignoresmasked_values_and_returns_zero_for_empty_rows():
    values = torch.tensor([[[1.0, 2.0], [100.0, 200.0]], [[7.0, 8.0], [9.0, 10.0]]])
    valid = torch.tensor([[1, 0], [0, 0]])

    result = mean(values, valid)

    assert torch.equal(result, torch.tensor([[1.0, 2.0], [0.0, 0.0]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def testattention_is_finite_and_zero_for_fullymasked_keys(dtype: torch.dtype):
    query = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
    key = torch.randn(2, 2, 4, dtype=dtype, requires_grad=True)
    valid = torch.tensor([[1, 0], [0, 0]])

    result = attention(query, key, valid)

    assert torch.isfinite(result).all()
    assert torch.equal(result[1], torch.zeros_like(result[1]))
    result.float().sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()


def testattention_normalizes_only_over_available_keys():
    query = torch.tensor([[[1.0, 0.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    result = attention(query, key, torch.tensor([[1, 0]]))

    assert torch.allclose(result, key[:, :1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def testmasked_softmax_does_not_exponentiate_extreme_invalid_scores(dtype: torch.dtype):
    scores = torch.tensor([[0.0, 1000.0]], dtype=dtype, requires_grad=True)

    weights = masked_softmax(scores, torch.tensor([[True, False]]))

    expected = torch.tensor([[1.0, 0.0]], dtype=dtype)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights, expected)
    weights.float().sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def testsequence_andfeatures_reject_unsupported_feature_inputs():
    with pytest.raises(ValueError, match="Expected \\[B, D\\] or \\[B, T, D\\]"):
        sequence(torch.ones(2, 3, 4, 5))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_dim = 4
            self.input_dims = {"audio": 4}
            self.proj = torch.nn.ModuleDict({"audio": torch.nn.Linear(4, 4)})

    batch = Batch(inputs={"audio": torch.ones(2, 3, 5)}, targets=None)
    with pytest.raises(ValueError, match="configured width"):
        features(Model(), batch, ("audio",))


def test_fusion_utilities_validate_shapes_and_configuration_errors():
    with pytest.raises(ValueError, match="normalized features"):
        mask(Batch(inputs={}, targets=None), "audio", torch.ones(2, 4))

    with pytest.raises(ValueError, match="temporal width"):
        mask(
            Batch(inputs={}, targets=None, masks={"audiomask": torch.ones(2, 2)}),
            "audio",
            torch.ones(2, 3, 4),
        )

    with pytest.raises(ValueError, match="batch dimension"):
        mask(
            Batch(inputs={}, targets=None, masks={"audiomask": torch.ones(2, 3)}),
            "audio",
            torch.ones(3, 3, 4),
        )

    one_step = Batch(
        inputs={},
        targets=None,
        masks={"sequencemask": torch.tensor([1, 0])},
    )
    assert torch.equal(mask(one_step, "audio", torch.ones(2, 1, 4)), torch.tensor([[True], [False]]))

    with pytest.raises(ValueError, match="sequencemask has batch dimension"):
        mask(
            Batch(inputs={}, targets=None, masks={"sequencemask": torch.ones(3, 2)}),
            "audio",
            torch.ones(2, 2, 4),
        )

    with pytest.raises(ValueError, match="sequencemask must"):
        mask(
            Batch(inputs={}, targets=None, masks={"sequencemask": torch.ones(2, 2, 1)}),
            "audio",
            torch.ones(2, 2, 4),
        )

    with pytest.raises(ValueError, match="Expected \[B, D\]"):
        sequence([1, 2, 3])
    with pytest.raises(ValueError, match="values with shape"):
        mean(torch.ones(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError, match="validity mask"):
        mean(torch.ones(2, 3, 4), torch.ones(2, 2))
    with pytest.raises(ValueError, match="at least two dimensions"):
        from chimera_ml.models.utils import masked_softmax

        masked_softmax(torch.ones(2), torch.ones(2, 1))
    with pytest.raises(ValueError, match="Expected validity mask"):
        masked_softmax(torch.ones(2, 3), torch.ones(2, 2))

    with pytest.raises(ValueError, match="query and key"):
        attention(torch.ones(2, 3), torch.ones(2, 3, 4), torch.ones(2, 3))
    with pytest.raises(ValueError, match="matching batch"):
        attention(torch.ones(2, 3, 4), torch.ones(1, 3, 4), torch.ones(1, 3))
    with pytest.raises(ValueError, match="key validity"):
        attention(torch.ones(2, 3, 4), torch.ones(2, 3, 4), torch.ones(2, 2))

    class BatchMismatchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_dim = 4
            self.input_dims = {"a": 4, "b": 4}
            self.proj = torch.nn.ModuleDict({"a": torch.nn.Linear(4, 4), "b": torch.nn.Linear(4, 4)})

    with pytest.raises(ValueError, match="batch dimension"):
        features(
            BatchMismatchModel(),
            Batch(inputs={"a": torch.ones(2, 1, 4), "b": torch.ones(3, 1, 4)}, targets=None),
            ("a", "b"),
        )

    class InvalidProjectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_dim = 4
            self.input_dims = {"audio": 4}
            self.proj = torch.nn.ModuleDict({"audio": torch.nn.Flatten()})

    with pytest.raises(ValueError, match="Projection for 'audio'"):
        features(InvalidProjectionModel(), Batch(inputs={"audio": torch.ones(2, 1, 4)}, targets=None), ("audio",))

    with pytest.raises(ValueError, match="No configured modalities"):
        features(BatchMismatchModel(), Batch(inputs={}, targets=None), ("a",))

    with pytest.raises(ValueError, match="shared_dim"):
        from chimera_ml.models.utils import projections

        projections(("audio",), {}, 0)
    with pytest.raises(ValueError, match="input_dims values"):
        projections(("audio",), {"audio": 0}, 4)


def testfeatures_reads_direct_batch_inputs_and_validates_declared_width():
    class DirectModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_dim = 3
            self.input_dims = {"audio": 4}
            self.proj = torch.nn.ModuleDict({"audio": torch.nn.Linear(4, 3)})

    model = DirectModel()
    batch = Batch(inputs={"audio": torch.ones(2, 4)}, targets=None)

    result = features(model, batch, ("audio",))

    assert result["audio"][0].shape == (2, 1, 3)
    with pytest.raises(ValueError, match="configured width"):
        features(model, Batch(inputs={"audio": torch.ones(2, 5)}, targets=None), ("audio",))


def testmask_prefers_dedicated_temporalmask_over_legacymasks():
    x = torch.ones(2, 3, 4)
    batch = Batch(
        inputs={},
        targets=None,
        masks={
            "audiomask": torch.ones(2),
            "audio_temporalmask": torch.tensor([[True, False, True], [False, True, False]]),
            "sequencemask": torch.zeros(2, 3, dtype=torch.bool),
        },
    )

    result = mask(batch, "audio", x)

    assert torch.equal(result, batch.masks["audio_temporalmask"])


def testmask_rejects_wrong_dedicated_temporalmask_shape():
    x = torch.ones(2, 3, 4)
    batch = Batch(inputs={}, targets=None, masks={"audio_temporalmask": torch.ones(2, 2)})

    with pytest.raises(ValueError, match="Dedicated temporal mask"):
        mask(batch, "audio", x)


def testmasked_softmax_returns_empty_weights_for_zero_length_keys():
    scores = torch.empty(2, 3, 0, requires_grad=True)
    valid = torch.zeros(2, 0, dtype=torch.bool)

    weights = masked_softmax(scores, valid)

    assert weights.shape == scores.shape
    assert weights.numel() == 0
    weights.sum().backward()
    assert scores.grad is not None and scores.grad.shape == scores.shape


def testattention_supports_zero_length_keysequences():
    query = torch.randn(2, 3, 4, requires_grad=True)
    key = torch.empty(2, 0, 4, requires_grad=True)
    valid = torch.zeros(2, 0, dtype=torch.bool)

    result = attention(query, key, valid)

    assert result.shape == query.shape
    assert torch.equal(result, torch.zeros_like(result))
    result.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
