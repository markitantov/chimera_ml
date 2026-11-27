from pathlib import Path

import pytest
import torch

import chimera_ml.models  # noqa: F401
from chimera_ml.core.batch import Batch
from chimera_ml.core.config import ExperimentConfig
from chimera_ml.training.builders import build_model

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fusion_yaml"


def _batch_for(model_name: str) -> Batch:
    if model_name in {"tfs_fusion_model"}:
        inputs = {
            "text": torch.randn(2, 3, 3),
            "audio": torch.randn(2, 2, 5),
            "video": torch.randn(2, 4, 7),
        }
    elif model_name == "raav_fusion_model":
        inputs = {
            "frame": torch.randn(2, 3, 3),
            "context": torch.randn(2, 2, 5),
        }
    else:
        inputs = {
            "text": torch.randn(2, 1, 3),
            "audio": torch.randn(2, 2, 5),
        }
    return Batch(inputs=inputs, targets=None)


@pytest.mark.parametrize(
    ("config_name", "expected_shape"),
    [
        ("tfs.yaml", (2, 2)),
        ("raav.yaml", (2, 3, 2)),
        ("tomar.yaml", (2, 2)),
    ],
)
def test_fusion_models_load_from_yaml_build_and_forward(config_name: str, expected_shape: tuple[int, ...]):
    config = ExperimentConfig.from_yaml(str(FIXTURE_DIR / config_name))
    model_config = config.section("model")

    model = build_model(model_config)
    model.eval()
    with torch.no_grad():
        output = model(_batch_for(model_config["name"]))

    assert output.preds.shape == expected_shape
    assert torch.isfinite(output.preds).all()
