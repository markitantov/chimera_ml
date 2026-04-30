import torch


def resolve_inference_device(device: str) -> str:
    """Resolve requested inference device to a concrete runtime value."""
    requested = device.strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda.")

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return requested
