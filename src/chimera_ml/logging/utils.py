import re
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo


_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def local_datetime_tag(
    *,
    include_time: bool = True,
    fmt: str | None = None,
    timezone: Optional[str] = None,
) -> str:
    """Local date/time tag suitable for experiment/run naming.

    By default includes both date and time to avoid collisions.
    """
    if timezone:
        dt = datetime.now(tz=ZoneInfo(timezone))
    else:
        dt = datetime.now().astimezone()
    if fmt:
        return dt.strftime(fmt)
    return dt.strftime("%Y-%m-%d_%H-%M") if include_time else dt.strftime("%Y-%m-%d")


def short_hash(text: str, n: int = 8) -> str:
    return sha1(text.encode("utf-8")).hexdigest()[:n]


def generate_run_name(
    config_path: Optional[str] = None,
    model_name: Optional[str] = None,
    suffix: Optional[str] = None,
    include_time: bool = True,
    datetime_format: Optional[str] = None,
    timezone: Optional[str] = None,
) -> str:
    """Create a human-readable, unique MLflow run name.
    """
    parts = []

    if config_path:
        parts.append(Path(config_path).stem)

    parts.append(local_datetime_tag(include_time=include_time, fmt=datetime_format, timezone=timezone))

    if model_name:
        parts.append(model_name)

    if suffix:
        parts.append(str(suffix))

    # Add a short stable hash of the config contents (if available)
    if config_path:
        try:
            cfg_text = Path(config_path).read_text(encoding="utf-8")
            parts.append(short_hash(cfg_text))
        except Exception:
            pass

    name = "_".join([p for p in parts if p])
    return name