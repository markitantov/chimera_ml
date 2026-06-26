import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chimera_ml.logging.utils import local_datetime_tag, short_hash

_SKIP_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
_SKIP_SUFFIX = {".pyc", ".pyo"}


def build_sweep_identifiers(
    *,
    sweep_name: str | None,
    sweep_config_text: str,
    timezone: str | None = None,
) -> tuple[str, str, str, str]:
    label = (sweep_name or "sweep").strip() or "sweep"
    date_tag = local_datetime_tag(fmt="%y%m%d-%H%M", timezone=timezone)
    started_at = local_datetime_tag(fmt="%Y-%m-%d_%H-%M-%S", timezone=timezone)
    hash_time = local_datetime_tag(fmt="%Y-%m-%d_%H-%M-%S-%f", timezone=timezone)
    short_id = short_hash(f"{sweep_config_text}\n{hash_time}", n=4)
    sweep_id = f"{label}-{date_tag}-{short_id}"
    return label, short_id, sweep_id, started_at


def resolve_sweep_log_root(cfg: Any) -> Path:
    logger_cfg = cfg.section("logging", name="console_file_logger")
    params = logger_cfg.get("params", {}) if logger_cfg else {}
    if isinstance(params, Mapping) and params.get("log_path"):
        return Path(params["log_path"])

    return Path("logs")


def zip_sources(zip_path: Path, base_dir: Path, include: list[str]) -> None:
    """Zip selected folders/files under base_dir, skipping caches/pyc."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for inc in include:
            inc_path = (base_dir / inc).resolve()
            if not inc_path.exists():
                continue

            if inc_path.is_file():
                if inc_path.suffix in _SKIP_SUFFIX:
                    continue

                zf.write(inc_path, str(inc_path.relative_to(base_dir)))
                continue

            for p in inc_path.rglob("*"):
                if not p.is_file():
                    continue

                if p.suffix in _SKIP_SUFFIX:
                    continue

                if any(part in _SKIP_DIRS for part in p.parts):
                    continue

                zf.write(p, str(p.relative_to(base_dir)))
