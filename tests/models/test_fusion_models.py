import pytest
import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.fusion import (
    FeatureFusionModel,
    PredictionFusionModel,
    feature_fusion_model,
    prediction_fusion_model,
)
from chimera_ml.models.gated_prediction_fusion import (
    GatedPredictionFusionModel,
    gated_prediction_fusion_model,
)
from chimera_ml.models.gating import GatedFusionModel, gated_fusion_model


class _ConstantSubmodel(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.register_buffer("_output", output)

    def forward(self, batch: Batch) -> ModelOutput:
        return ModelOutput(preds=self._output.clone())


def test_feature_fusion_forward_returns_preds_and_aux():
    encoders = {
        "text": nn.Linear(3, 2, bias=False),
        "audio": nn.Linear(4, 2, bias=False),
    }
    model = FeatureFusionModel(encoders=encoders, head=nn.Linear(4, 1), dropout=0.0, use_mask=True)

    batch = Batch(
        inputs={"text": torch.randn(2, 3), "audio": torch.randn(2, 4)},
        targets=None,
        masks={"text_mask": torch.tensor([1.0, 0.0]), "audio_mask": torch.tensor([1.0, 1.0])},
    )

    out = model(batch)

    assert out.preds.shape == (2, 1)
    assert out.aux is not None
    assert "emb_text" in out.aux
    assert "emb_audio" in out.aux


def test_feature_fusion_raises_when_no_known_modalities():
    model = FeatureFusionModel(encoders={"text": nn.Linear(2, 2)}, head=nn.Linear(2, 1))
    batch = Batch(inputs={"audio": torch.randn(2, 2)}, targets=None)

    with pytest.raises(ValueError, match="No modalities provided"):
        model(batch)


def test_prediction_fusion_mean_sum_weighted_and_errors():
    preds_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    preds_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    submodels = {"a": _ConstantSubmodel(preds_a), "b": _ConstantSubmodel(preds_b)}
    batch = Batch(inputs={"a": torch.zeros(2, 1), "b": torch.zeros(2, 1)}, targets=None)

    mean_model = PredictionFusionModel(submodels=submodels, fusion="mean")
    out_mean = mean_model(batch)
    assert torch.allclose(out_mean.preds, (preds_a + preds_b) / 2.0)

    sum_model = PredictionFusionModel(submodels=submodels, fusion="sum")
    out_sum = sum_model(batch)
    assert torch.allclose(out_sum.preds, preds_a + preds_b)

    weighted_model = PredictionFusionModel(
        submodels=submodels,
        fusion="weighted",
        weights={"a": 1.0, "b": 3.0},
    )
    out_weighted = weighted_model(batch)
    expected = (preds_a * 1.0 + preds_b * 3.0) / 4.0
    assert torch.allclose(out_weighted.preds, expected)

    with pytest.raises(ValueError, match="Unknown fusion strategy"):
        PredictionFusionModel(submodels=submodels, fusion="bad")(batch)

    with pytest.raises(ValueError, match="No modalities provided"):
        PredictionFusionModel(submodels=submodels, fusion="mean")(Batch(inputs={}, targets=None))


def test_gated_fusion_forward_and_projection_mismatch():
    encoders = {"text": nn.Linear(3, 2), "audio": nn.Linear(4, 2)}
    model = GatedFusionModel(
        encoders=encoders,
        head=nn.Linear(2, 1),
        shared_dim=2,
        gate_hidden=4,
        use_mask=True,
    )
    batch = Batch(
        inputs={"text": torch.randn(2, 3), "audio": torch.randn(2, 4)},
        targets=None,
        masks={"text_mask": torch.tensor([1.0, 1.0]), "audio_mask": torch.tensor([1.0, 0.0])},
    )

    out = model(batch)
    assert out.preds.shape == (2, 1)
    assert out.aux is not None
    assert out.aux["gates"].shape == (2, 2)

    mismatch = GatedFusionModel(
        encoders={"text": nn.Linear(3, 3)},
        head=nn.Linear(2, 1),
        shared_dim=2,
        gate_hidden=4,
    )
    with pytest.raises(ValueError, match="Embedding dim mismatch"):
        mismatch(Batch(inputs={"text": torch.randn(2, 3)}, targets=None))

    mismatch.set_projection("text", nn.Linear(3, 2))
    out_fixed = mismatch(Batch(inputs={"text": torch.randn(2, 3)}, targets=None))
    assert out_fixed.preds.shape == (2, 1)

    with pytest.raises(ValueError, match="No modalities provided"):
        model(Batch(inputs={}, targets=None))


def test_gated_prediction_fusion_forward_and_errors():
    logits_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits_b = torch.tensor([[0.5, 0.5], [2.0, 1.0]])
    submodels = {"a": _ConstantSubmodel(logits_a), "b": _ConstantSubmodel(logits_b)}
    model = GatedPredictionFusionModel(submodels=submodels, num_classes=2, gate_hidden=4, use_mask=True)

    batch = Batch(
        inputs={"a": torch.zeros(2, 1), "b": torch.zeros(2, 1)},
        targets=None,
        masks={"a_mask": torch.tensor([1.0, 1.0]), "b_mask": torch.tensor([1.0, 0.0])},
    )

    out = model(batch)
    assert out.preds.shape == (2, 2)
    assert out.aux is not None
    assert out.aux["gates"].shape == (2, 2)

    wrong = GatedPredictionFusionModel(submodels={"a": _ConstantSubmodel(torch.randn(2, 3))}, num_classes=2)
    with pytest.raises(ValueError, match="must output"):
        wrong(Batch(inputs={"a": torch.zeros(2, 1)}, targets=None))

    with pytest.raises(ValueError, match="No modalities provided"):
        model(Batch(inputs={}, targets=None))


def test_model_factories_and_registry_entries_work():
    f = MODELS.get("feature_fusion_model")
    p = MODELS.get("prediction_fusion_model")
    g = MODELS.get("gated_fusion_model")
    gp = MODELS.get("gated_prediction_fusion_model")

    assert callable(f)
    assert callable(p)
    assert callable(g)
    assert callable(gp)

    feat = feature_fusion_model(encoders={"x": nn.Linear(2, 2)}, head=nn.Linear(2, 1))
    pred = prediction_fusion_model(submodels={"x": _ConstantSubmodel(torch.randn(2, 2))})
    gate = gated_fusion_model(encoders={"x": nn.Linear(2, 2)}, head=nn.Linear(2, 1), shared_dim=2)
    gate_pred = gated_prediction_fusion_model(submodels={"x": _ConstantSubmodel(torch.randn(2, 2))}, num_classes=2)

    assert isinstance(feat, FeatureFusionModel)
    assert isinstance(pred, PredictionFusionModel)
    assert isinstance(gate, GatedFusionModel)
    assert isinstance(gate_pred, GatedPredictionFusionModel)
