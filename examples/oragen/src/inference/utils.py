from pathlib import Path

from chimera_ml.inference import InferenceContext


def resolve_checkpoint_path(ctx: InferenceContext, checkpoint_key: str) -> str:
    checkpoints = ctx.get_artifact("checkpoints", {})
    if not isinstance(checkpoints, dict):
        raise TypeError("Inference artifact 'checkpoints' must be a dict when present.")

    resolved_path = checkpoints.get(checkpoint_key)
    if resolved_path is None:
        raise FileNotFoundError(f"Missing checkpoint key '{checkpoint_key}' in inference artifact 'checkpoints' ")

    path = Path(str(resolved_path)).expanduser()
    if path.is_file():
        return str(path.resolve())

    if path.exists():
        raise FileNotFoundError(f"Resolved checkpoint for key '{checkpoint_key}' is not a file: {path} ")

    raise FileNotFoundError(f"Resolved checkpoint for key '{checkpoint_key}' does not exist: {path} ")
