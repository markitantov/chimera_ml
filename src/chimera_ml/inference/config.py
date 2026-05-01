from dataclasses import dataclass
from typing import Any

from chimera_ml.core.config import ExperimentConfig


@dataclass
class InferenceConfig(ExperimentConfig):
    """Small wrapper around inference YAML config."""

    @property
    def pipeline_name(self) -> str:
        pipeline_cfg = self.raw.get("pipeline", {})
        if not isinstance(pipeline_cfg, dict):
            return "inference_pipeline"

        name = pipeline_cfg.get("name")
        return str(name) if name else "inference_pipeline"

    @property
    def parallel(self) -> bool:
        pipeline_cfg = self.raw.get("pipeline", {})
        if not isinstance(pipeline_cfg, dict):
            return False

        value = pipeline_cfg.get("parallel", False)
        if not isinstance(value, bool):
            raise TypeError("Inference config field 'pipeline.parallel' must be a boolean.")

        return value

    @property
    def steps(self) -> list[dict[str, Any]]:
        steps_cfg = self.raw.get("steps", [])
        if not isinstance(steps_cfg, list):
            raise TypeError("Inference config section 'steps' must be a list.")

        return steps_cfg

    def runtime_device(self, default: str = "auto") -> str:
        runtime_cfg = self.raw.get("runtime", {})
        if not isinstance(runtime_cfg, dict):
            return default

        device = runtime_cfg.get("device", default)
        return str(device)
