import requests
from pathlib import Path
from contextlib import suppress
from urllib.parse import urlparse


def load_model(
    model_url: str, 
    cache_dir: str, 
    force_reload: bool = False
) -> str | None:

    file_name = Path(urlparse(model_url).path).name
    file_path = Path(cache_dir) / file_name

    if file_path.exists() and not force_reload:
        return str(file_path)

    with suppress(Exception), requests.get(model_url, stream=True) as response:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        return str(file_path)

    return None
