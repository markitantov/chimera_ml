import importlib
import warnings

_MODULES_TO_REGISTER: tuple[str, ...] = (
    "audio.data.audio_datamodule"
    "audio.models.audio_models",
    "audio.models.audio_sota_model",
    "image.models.image_models",
    "fusion.data.fusion_features_datamodule",
    "fusion.models.fusion_models",
    "common.losses",
    "common.metrics",
)


def register() -> None:
    for module_name in _MODULES_TO_REGISTER:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Optional runtime dependency is missing (e.g. torchaudio for audio datamodules).
            # Keep plugin registration usable for available components (e.g. fusion pipeline).
            if exc.name in {"torchaudio", "torchvision", "transformers", "pandas"}:
                continue
            warnings.warn(f"Failed to import '{module_name}': {exc}", stacklevel=2)
        except Exception as exc:
            warnings.warn(f"Failed to import '{module_name}': {exc}", stacklevel=2)
