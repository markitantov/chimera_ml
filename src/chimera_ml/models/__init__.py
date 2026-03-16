# factories registered in MODELS registry
from chimera_ml.models.base import BaseModel

from chimera_ml.models.fusion import feature_fusion_model, prediction_fusion_model
from chimera_ml.models.gating import gated_fusion_model
from chimera_ml.models.gated_prediction_fusion import gated_prediction_fusion_model

__all__ = [
    "BaseModel",
    "feature_fusion_model",
    "prediction_fusion_model",
    "gated_fusion_model",
    "gated_prediction_fusion_model",
]
