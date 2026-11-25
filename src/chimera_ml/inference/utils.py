"""Shared inference utilities."""

import torch


def resolve_inference_device(device: str) -> str:
    requested = device.strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda.")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else requested
