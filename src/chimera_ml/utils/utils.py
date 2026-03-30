import zipfile
from pathlib import Path

_SKIP_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
_SKIP_SUFFIX = {".pyc", ".pyo"}


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
