import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference.context import InferenceContext


def _is_remote_checkpoint(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _resolve_cache_dir(work_dir: Path, cache_dir: str | Path) -> Path:
    cache_path = Path(cache_dir).expanduser()
    if not cache_path.is_absolute():
        cache_path = work_dir / cache_path

    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _checkpoint_filename(name: str, ref: str) -> str:
    parsed = urlparse(ref)
    filename = Path(parsed.path).name if parsed.path else ""
    if filename:
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        digest = hashlib.sha256(ref.encode("utf-8")).hexdigest()[:12]
        return f"{stem}-{digest}{suffix}"

    local_name = Path(ref).expanduser().name
    if local_name:
        return local_name

    digest = hashlib.sha256(ref.encode("utf-8")).hexdigest()[:12]
    return f"{name}-{digest}.pt"


def _require_existing_file(path: Path, *, description: str) -> str:
    if path.is_file():
        return str(path.resolve())

    if path.exists():
        raise FileNotFoundError(f"{description} is not a file: {path}")

    raise FileNotFoundError(f"{description} not found: {path}")


@dataclass
class ResolveCheckpointsStep:
    checkpoints: dict[str, str]
    cache_dir: str = "checkpoints"
    force_download: bool = False
    chunk_size: int = 8192

    def run(self, ctx: InferenceContext) -> InferenceContext:
        existing = ctx.get_artifact("checkpoints", {})
        if not isinstance(existing, dict):
            raise TypeError("Inference artifact 'checkpoints' must be a dict when present.")

        resolved = dict(existing)
        cache_dir = _resolve_cache_dir(ctx.work_dir, self.cache_dir)
        for name, ref in self.checkpoints.items():
            resolved[name] = self._resolve_one(name=name, ref=ref, cache_dir=cache_dir)

        ctx.set_artifact("checkpoints", resolved)
        ctx.set_artifact("cache_dir", cache_dir)
        return ctx

    def _resolve_one(self, *, name: str, ref: str, cache_dir: Path) -> str:
        local_path = Path(ref).expanduser()
        if local_path.exists():
            resolved_path = _require_existing_file(local_path, description=f"Checkpoint '{name}'")
            print(f"[inference] Checkpoint '{name}': using local file {resolved_path}")
            return resolved_path

        if not _is_remote_checkpoint(ref):
            message = f"Checkpoint '{name}' not found: {ref}"
            print(f"[inference] {message}")
            raise FileNotFoundError(message)

        target_path = cache_dir / _checkpoint_filename(name, ref)
        if target_path.exists() and not self.force_download:
            resolved_path = _require_existing_file(target_path, description=f"Cached checkpoint '{name}'")
            print(f"[inference] Checkpoint '{name}': using cached file {resolved_path}")
            return resolved_path

        if target_path.exists() and not target_path.is_file():
            raise FileNotFoundError(f"Cached checkpoint '{name}' is not a file: {target_path}")

        print(f"[inference] Checkpoint '{name}': downloading {ref}")
        print(f"[inference] Checkpoint '{name}': saving to {target_path.resolve()}")
        return self._download_checkpoint(name=name, ref=ref, target_path=target_path)

    def _download_checkpoint(self, *, name: str, ref: str, target_path: Path) -> str:
        partial_path = target_path.with_suffix(target_path.suffix + ".part")
        partial_path.unlink(missing_ok=True)

        try:
            with requests.get(ref, stream=True, timeout=30) as response:
                response.raise_for_status()
                total = self._parse_total_bytes(response.headers.get("content-length"))
                with (
                    partial_path.open("wb") as file_obj,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=f"[inference] Downloading {name}",
                    ) as progress,
                ):
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if not chunk:
                            continue

                        file_obj.write(chunk)
                        progress.update(len(chunk))

            partial_path.replace(target_path)
        except requests.RequestException as exc:
            partial_path.unlink(missing_ok=True)
            message = f"Failed to download checkpoint '{name}' from {ref}: {exc}"
            print(f"[inference] {message}")
            raise FileNotFoundError(message) from exc
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

        resolved_path = str(target_path.resolve())
        print(f"[inference] Checkpoint '{name}': saved to {resolved_path}")
        return resolved_path

    @staticmethod
    def _parse_total_bytes(value: str | None) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None


@INFERENCE_STEPS.register("resolve_checkpoints_step")
def resolve_checkpoints_step(**params: Any) -> ResolveCheckpointsStep:
    return ResolveCheckpointsStep(**params)
