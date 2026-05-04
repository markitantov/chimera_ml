import argparse
from pathlib import Path
from typing import Any

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a compact inference checkpoint from a training checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the training checkpoint (.pt).",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to save the inference checkpoint. Defaults next to the source checkpoint.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert floating-point model weights to float16 before saving.",
    )
    return parser.parse_args()


def _resolve_output_path(checkpoint_path: Path, output_path: str | None, fp16: bool) -> Path:
    if output_path:
        return Path(output_path)

    suffix = "_inference_fp16.pt" if fp16 else "_inference.pt"
    return checkpoint_path.with_name(f"{checkpoint_path.stem}{suffix}")


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")

    return state_dict


def _convert_state_dict(
    state_dict: dict[str, Any],
    *,
    fp16: bool,
) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if fp16 and torch.is_floating_point(tensor):
                tensor = tensor.half()
            converted[key] = tensor
        else:
            converted[key] = value
    return converted


def _state_dict_size_mb(state_dict: dict[str, Any]) -> float:
    total_bytes = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total_bytes += value.numel() * value.element_size()

    return total_bytes / 1024 / 1024


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    output_path = _resolve_output_path(checkpoint_path, args.output_path, args.fp16)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[export] Loading checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _extract_state_dict(payload)
    converted_state_dict = _convert_state_dict(state_dict, fp16=bool(args.fp16))

    export_payload = {
        "model_state_dict": converted_state_dict,
        "source_checkpoint": str(checkpoint_path),
        "precision": "fp16" if args.fp16 else "fp32",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_payload, output_path)

    print(f"[export] Saved inference checkpoint: {output_path}")
    print(f"[export] precision={export_payload['precision']}")
    print(f"[export] tensors={len(converted_state_dict)}")
    print(f"[export] approx_model_size_mb={_state_dict_size_mb(converted_state_dict):.1f}")


if __name__ == "__main__":
    main()
