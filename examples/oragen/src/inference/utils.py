from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

import requests


def load_model(model_url: str, cache_dir: str | Path, force_reload: bool = False) -> str | None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = Path(model_url).expanduser()
    if local_path.exists():
        return str(local_path)

    parsed = urlparse(model_url)
    if parsed.scheme in {"http", "https"}:
        file_name = Path(parsed.path).name or local_path.name
        file_path = cache_dir / file_name
        if file_path.exists() and not force_reload:
            return str(file_path)

        with suppress(Exception), requests.get(model_url, stream=True) as response:
            response.raise_for_status()
            with file_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            return str(file_path)

    raise FileNotFoundError(f"Model file not found: {model_url}")
