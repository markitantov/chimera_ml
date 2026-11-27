"""MODELS package.

Built-in models are exposed and registered by default.
Importing this package ensures registry decorators are executed.
"""

from chimera_ml.models.base import BaseModel
from chimera_ml.models.raav_fusion_model import RAAVFusionModel, raav_fusion_model
from chimera_ml.models.tfs_fusion_model import TFSFusionModel, tfs_fusion_model
from chimera_ml.models.tomar_fusion import TOMARFusionModel, tomar_fusion_model

__all__ = [
    "BaseModel",
    "RAAVFusionModel",
    "TFSFusionModel",
    "TOMARFusionModel",
    "raav_fusion_model",
    "tfs_fusion_model",
    "tomar_fusion_model",
]
