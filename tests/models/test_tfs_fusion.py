import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.models.common_layers import PositionalEncoding, StatPool, TransformerCrossAttention
from chimera_ml.models.tfs_fusion_model import TFSFusionModel, normalize_groups, tfs_fusion_model


def batch() -> Batch:
    return Batch(
        inputs={
            "text": torch.randn(2, 3, 4),
            "audio": torch.randn(2, 3, 5),
            "video": torch.randn(2, 3, 6),
        },
        targets=None,
        masks={
            "text_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
            "audio_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "video_mask": torch.tensor([[1, 0, 0], [1, 1, 1]]),
        },
    )


def test_tfs_fuses_all_directed_pairs_and_groups():
    model = TFSFusionModel(
        input_dims={"text": 4, "audio": 5, "video": 6},
        shared_dim=4,
        output_dim=2,
    )

    output = model(batch())

    assert output.preds.shape == (2, 2)
    assert output.aux is not None
    assert output.aux["fused"].shape == (2, 4)
    assert output.aux["pair_text->audio"].shape == (2, 3, 4)
    assert output.aux["pair_audio->text"].shape == (2, 3, 4)
    assert output.aux["group_weights"].shape == (2, 2)


def test_tfs_supports_explicit_groups_and_missing_modalities():
    model = TFSFusionModel(
        input_dims={"text": 3, "audio": 3, "video": 3},
        shared_dim=4,
        output_dim=2,
        groups=((("text", "audio"),), (("audio", "text"),)),
    )

    output = model(Batch(inputs={"text": torch.randn(2, 3)}, targets=None))

    assert output.preds.shape == (2, 2)
    assert "pair_text->audio" not in output.aux


def test_tfs_rejects_invalid_groups():
    with pytest.raises(ValueError, match="occurs more than once"):
        normalize_groups(
            ("text", "audio"),
            ((("text", "audio"),), (("text", "audio"),)),
        )


def test_tfs_factory_is_registered():
    assert MODELS.get("tfs_fusion_model") is tfs_fusion_model
    assert isinstance(
        tfs_fusion_model(input_dims={"text": 3, "audio": 3}, shared_dim=4, output_dim=2),
        TFSFusionModel,
    )


def test_stat_pool_returns_mean_and_population_std():
    values = torch.tensor([[[2.0], [4.0], [6.0]], [[1.0], [1.0], [1.0]]])
    valid = torch.ones(2, 3, dtype=torch.bool)

    pooled = StatPool()(values, valid)

    assert torch.allclose(pooled, torch.tensor([[4.0, 1.632993], [1.0, 0.0]]), atol=1e-5)


def test_transformer_cross_attention_handles_masks_and_empty_sequences():
    block = TransformerCrossAttention(4, dropout=0.0).eval()
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 2, 4)

    output, weights = block(
        query,
        key,
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([[1, 1], [0, 0]], dtype=torch.bool),
    )

    assert output.shape == query.shape
    assert weights.shape == (2, 3, 2)
    assert torch.isfinite(output).all()
    assert torch.equal(
        block(
            torch.empty(2, 0, 4),
            key,
            torch.zeros(2, 0, dtype=torch.bool),
            torch.ones(2, 2, dtype=torch.bool),
        )[0],
        torch.empty(2, 0, 4),
    )


def test_positional_encoding_validates_sequence_length():
    encoding = PositionalEncoding(dim=4, dropout=0.0, max_len=3).eval()

    assert encoding(torch.zeros(2, 3, 4)).shape == (2, 3, 4)
    with pytest.raises(ValueError, match="max_len 3"):
        encoding(torch.zeros(2, 4, 4))
