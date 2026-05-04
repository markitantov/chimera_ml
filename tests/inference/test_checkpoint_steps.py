from pathlib import Path

import pytest
import requests

from chimera_ml.inference import InferenceConfig, InferenceContext, build_inference_pipeline
from chimera_ml.inference.steps import checkpoint_steps
from chimera_ml.inference.steps.checkpoint_steps import ResolveCheckpointsStep, _checkpoint_filename


def _make_ctx(tmp_path, *, artifacts=None) -> InferenceContext:
    return InferenceContext(
        input_path=tmp_path / "input.mp4",
        work_dir=tmp_path,
        device="cpu",
        config={},
        artifacts=artifacts or {},
    )


class _FakeResponse:
    def __init__(self, *, chunks: list[bytes], headers: dict[str, str] | None = None, error: Exception | None = None):
        self._chunks = chunks
        self.headers = headers or {}
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield from self._chunks


class _FakeProgress:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, amount: int) -> None:
        self.updates.append(amount)


def test_builtin_resolve_checkpoints_step_uses_local_path_and_merges_artifacts(tmp_path, capsys):
    local_checkpoint = tmp_path / "audio_model.pt"
    local_checkpoint.write_bytes(b"local")

    pipeline = build_inference_pipeline(
        InferenceConfig(
            {
                "steps": [
                    {
                        "name": "resolve_checkpoints_step",
                        "params": {
                            "checkpoints": {"audio": str(local_checkpoint)},
                            "cache_dir": "model_cache",
                        },
                    }
                ]
            }
        )
    )

    ctx = pipeline.run(_make_ctx(tmp_path, artifacts={"checkpoints": {"existing": "/tmp/existing.pt"}}))

    assert ctx.get_artifact("checkpoints") == {
        "existing": "/tmp/existing.pt",
        "audio": str(local_checkpoint.resolve()),
    }
    assert "Checkpoint 'audio': using local file" in capsys.readouterr().out


def test_resolve_checkpoints_step_downloads_with_progress_and_saves_to_cache(tmp_path, monkeypatch, capsys):
    progress_instances: list[_FakeProgress] = []

    def _fake_get(url: str, *, stream: bool, timeout: int):
        assert url == "https://example.com/fusion.pt"
        assert stream is True
        assert timeout == 30
        return _FakeResponse(chunks=[b"abc", b"def"], headers={"content-length": "6"})

    def _fake_tqdm(**kwargs):
        progress = _FakeProgress(**kwargs)
        progress_instances.append(progress)
        return progress

    monkeypatch.setattr(checkpoint_steps.requests, "get", _fake_get)
    monkeypatch.setattr(checkpoint_steps, "tqdm", _fake_tqdm)

    step = ResolveCheckpointsStep(
        checkpoints={"fusion": "https://example.com/fusion.pt"},
        cache_dir="model_cache",
    )
    ctx = step.run(_make_ctx(tmp_path))

    target_path = tmp_path / "model_cache" / _checkpoint_filename("fusion", "https://example.com/fusion.pt")
    assert target_path.read_bytes() == b"abcdef"
    assert ctx.get_artifact("checkpoints") == {"fusion": str(target_path.resolve())}

    assert len(progress_instances) == 1
    assert progress_instances[0].kwargs["total"] == 6
    assert progress_instances[0].kwargs["desc"] == "[inference] Downloading fusion"
    assert progress_instances[0].updates == [3, 3]

    out = capsys.readouterr().out
    assert "Checkpoint 'fusion': downloading https://example.com/fusion.pt" in out
    assert f"Checkpoint 'fusion': saving to {target_path.resolve()}" in out
    assert f"Checkpoint 'fusion': saved to {target_path.resolve()}" in out


def test_resolve_checkpoints_step_uses_cached_file_without_redownloading(tmp_path, monkeypatch, capsys):
    cached_path = tmp_path / "model_cache" / _checkpoint_filename("fusion", "https://example.com/fusion.pt")
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"cached")

    def _unexpected_get(*args, **kwargs):
        raise AssertionError("requests.get should not be called for cached checkpoints")

    monkeypatch.setattr(checkpoint_steps.requests, "get", _unexpected_get)

    step = ResolveCheckpointsStep(
        checkpoints={"fusion": "https://example.com/fusion.pt"},
        cache_dir="model_cache",
    )
    ctx = step.run(_make_ctx(tmp_path))

    assert ctx.get_artifact("checkpoints") == {"fusion": str(cached_path.resolve())}
    assert "Checkpoint 'fusion': using cached file" in capsys.readouterr().out


def test_resolve_checkpoints_step_uses_distinct_cache_paths_for_same_basename_urls(tmp_path, monkeypatch):
    payloads = {
        "https://host-a.example/models/model.pt": [b"aaa"],
        "https://host-b.example/other/model.pt": [b"bbb"],
    }

    def _fake_get(url: str, *, stream: bool, timeout: int):
        assert stream is True
        assert timeout == 30
        return _FakeResponse(chunks=payloads[url], headers={"content-length": "3"})

    monkeypatch.setattr(checkpoint_steps.requests, "get", _fake_get)
    monkeypatch.setattr(checkpoint_steps, "tqdm", lambda **kwargs: _FakeProgress(**kwargs))

    step = ResolveCheckpointsStep(
        checkpoints={
            "first": "https://host-a.example/models/model.pt",
            "second": "https://host-b.example/other/model.pt",
        },
        cache_dir="model_cache",
    )
    ctx = step.run(_make_ctx(tmp_path))

    first_path = Path(ctx.get_artifact("checkpoints")["first"])
    second_path = Path(ctx.get_artifact("checkpoints")["second"])
    assert first_path != second_path
    assert first_path.name != second_path.name
    assert first_path.read_bytes() == b"aaa"
    assert second_path.read_bytes() == b"bbb"


def test_resolve_checkpoints_step_rejects_directory_local_checkpoint(tmp_path):
    local_dir = tmp_path / "audio_model.pt"
    local_dir.mkdir()

    step = ResolveCheckpointsStep(checkpoints={"audio": str(local_dir)})

    with pytest.raises(FileNotFoundError, match=r"Checkpoint 'audio' is not a file"):
        step.run(_make_ctx(tmp_path))


def test_resolve_checkpoints_step_rejects_directory_cached_checkpoint(tmp_path):
    cached_dir = tmp_path / "model_cache" / _checkpoint_filename("fusion", "https://example.com/fusion.pt")
    cached_dir.mkdir(parents=True)

    step = ResolveCheckpointsStep(
        checkpoints={"fusion": "https://example.com/fusion.pt"},
        cache_dir="model_cache",
    )

    with pytest.raises(FileNotFoundError, match=r"Cached checkpoint 'fusion' is not a file"):
        step.run(_make_ctx(tmp_path))


def test_resolve_checkpoints_step_raises_for_missing_local_path(tmp_path, capsys):
    missing_path = tmp_path / "missing.pt"
    step = ResolveCheckpointsStep(checkpoints={"audio": str(missing_path)})

    with pytest.raises(FileNotFoundError, match=f"Checkpoint 'audio' not found: {missing_path}"):
        step.run(_make_ctx(tmp_path))

    assert f"Checkpoint 'audio' not found: {missing_path}" in capsys.readouterr().out


def test_resolve_checkpoints_step_reports_download_errors_and_cleans_partial_file(tmp_path, monkeypatch, capsys):
    def _failing_get(url: str, *, stream: bool, timeout: int):
        del url, stream, timeout
        raise requests.RequestException("network down")

    monkeypatch.setattr(checkpoint_steps.requests, "get", _failing_get)

    step = ResolveCheckpointsStep(
        checkpoints={"fusion": "https://example.com/fusion.pt"},
        cache_dir="model_cache",
    )

    with pytest.raises(FileNotFoundError, match="Failed to download checkpoint 'fusion'"):
        step.run(_make_ctx(tmp_path))

    partial_path = (
        tmp_path / "model_cache" / (_checkpoint_filename("fusion", "https://example.com/fusion.pt") + ".part")
    )
    assert not partial_path.exists()
    assert "Failed to download checkpoint 'fusion' from https://example.com/fusion.pt: network down" in (
        capsys.readouterr().out
    )
