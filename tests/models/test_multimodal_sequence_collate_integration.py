import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.data.multimodal_sequence_collate import MultimodalSequenceCollate
from chimera_ml.training.builders import build_model


def _samples(model_name: str, invalid_offset: float = 0.0) -> list[dict[str, object]]:
    if model_name in {"tfs_fusion_model", "tfs_dcmmmoe_fusion_model"}:
        return [
            {
                "inputs": {
                    "text": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                    "audio": torch.cat(
                        [
                            torch.arange(5, dtype=torch.float32).reshape(1, 5),
                            torch.full((1, 5), invalid_offset),
                        ]
                    ),
                    "video": torch.arange(7, dtype=torch.float32).reshape(1, 7),
                },
                "meta": {"masks": {"audio_temporal_mask": [True, False]}},
            },
            {
                "inputs": {
                    "text": torch.ones(1, 3),
                    "audio": torch.full((3, 5), 2.0),
                }
            },
        ]
    if model_name == "raav_fusion_model":
        return [
            {
                "inputs": {
                    "frame_a": torch.cat(
                        [
                            torch.arange(3, dtype=torch.float32).reshape(1, 3),
                            torch.full((1, 3), invalid_offset),
                        ]
                    ),
                    "frame_b": torch.arange(10, dtype=torch.float32).reshape(2, 5),
                    "context": torch.ones(1, 7),
                },
                "meta": {"masks": {"frame_a_temporal_mask": [True, False]}},
            },
            {
                "inputs": {
                    "frame_a": torch.full((3, 3), 2.0),
                    "frame_b": torch.full((3, 5), 3.0),
                }
            },
        ]
    return [
        {
            "inputs": {
                "text": torch.tensor([1.0, 2.0, 3.0]),
                "audio": torch.arange(10, dtype=torch.float32).reshape(2, 5),
            },
            "meta": {"masks": {"audio_temporal_mask": [True, False]}},
        },
        {"inputs": {"text": torch.tensor([2.0, 3.0, 4.0])}},
    ]


@pytest.mark.parametrize(
    ("model_name", "params", "expected_shape"),
    [
        (
            "tfs_fusion_model",
            {"input_dims": {"text": 3, "audio": 5, "video": 7}, "shared_dim": 4, "output_dim": 2},
            (2, 2),
        ),
        (
            "tfs_dcmmmoe_fusion_model",
            {
                "input_dims": {"text": 3, "audio": 5, "video": 7},
                "shared_dim": 4,
                "output_dim": 2,
                "gate_hidden": 3,
            },
            (2, 2),
        ),
        (
            "raav_fusion_model",
            {
                "input_dims": {"frame_a": 3, "frame_b": 5, "context": 7},
                "shared_dim": 4,
                "output_dim": 2,
                "frame_modalities": ["frame_a", "frame_b"],
                "context_modality": "context",
                "num_latents": 2,
                "gate_hidden": 3,
            },
            (2, 3, 2),
        ),
        (
            "tomar_fusion_model",
            {
                "input_dims": {"text": 3, "audio": 5},
                "shared_dim": 4,
                "anchor_modality": "text",
                "output_dim": 2,
                "gate_hidden": 3,
            },
            (2, 2),
        ),
    ],
)
def test_fusion_models_use_multimodal_sequence_masks(model_name, params, expected_shape):
    torch.manual_seed(7)
    model = build_model({"name": model_name, "params": params})
    model.eval()
    collate = MultimodalSequenceCollate(pad_value=17.0)
    batch = collate(_samples(model_name))
    changed_batch = collate(_samples(model_name, invalid_offset=1000.0))

    with torch.no_grad():
        output = model(batch)
        changed_output = model(changed_batch)

    assert isinstance(batch, Batch)
    assert output.preds.shape == expected_shape
    assert torch.isfinite(output.preds).all()
    assert torch.allclose(output.preds, changed_output.preds)

    model.zero_grad()
    gradient_output = model(batch)
    gradient_output.preds.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
