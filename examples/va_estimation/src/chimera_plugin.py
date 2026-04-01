from __future__ import annotations

import importlib
import warnings

_MODULES_TO_REGISTER: tuple[str, ...] = (
    "audio.data.audio_va_datamodule",
    "audio.models.wavlm_s2s_model",
    "audio.loss.audio_ccc_mse_loss",
    "audio.optimizers.adamw_two_group_optimizer",
    "audio.schedulers.named_reduceonplateau_scheduler",
    "audio.callbacks.audio_framewise_callback",
    "audio.callbacks.audio_windowwise_callback",
    "audio.callbacks.audio_unfreeze_backbone_callback",
    "metrics.va_ccc_metric",
    "fusion.data.fusion_va_datamodule",
    "fusion.models.fusion_model",
    "fusion.loss.fusion_ccc_mse_loss",
    "fusion.callbacks.fusion_framewise_callback",
)


def register() -> None:
    for module_name in _MODULES_TO_REGISTER:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Optional runtime dependency is missing (e.g. torchaudio for audio datamodules).
            # Keep plugin registration usable for available components (e.g. fusion pipeline).
            if exc.name in {"torchaudio"}:
                continue
            warnings.warn(f"Failed to import '{module_name}': {exc}", stacklevel=2)
        except Exception as exc:
            warnings.warn(f"Failed to import '{module_name}': {exc}", stacklevel=2)
