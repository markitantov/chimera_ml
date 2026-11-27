import chimera_ml.models as public_models
from chimera_ml.core.registry import MODELS


def test_public_fusion_classes_and_factories_are_exported():
    from chimera_ml.models import (
        RAAVFusionModel,
        TFSFusionModel,
        TOMARFusionModel,
        raav_fusion_model,
        tfs_fusion_model,
        tomar_fusion_model,
    )

    classes = (TFSFusionModel, RAAVFusionModel, TOMARFusionModel)
    factories = (tfs_fusion_model, raav_fusion_model, tomar_fusion_model)

    assert all(getattr(public_models, cls.__name__) is cls for cls in classes)
    assert all(name in public_models.__all__ for name in [cls.__name__ for cls in classes])
    assert all(factory.__name__ in public_models.__all__ for factory in factories)
    assert MODELS.get("tfs_fusion_model") is tfs_fusion_model
    assert MODELS.get("raav_fusion_model") is raav_fusion_model
    assert MODELS.get("tomar_fusion_model") is tomar_fusion_model
