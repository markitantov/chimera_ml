import zipfile
from pathlib import Path

from chimera_ml.utils.utils import zip_sources


def test_zip_sources_skips_caches_and_pyc(tmp_path: Path):
    base = tmp_path / "project"
    src = base / "src"
    pycache = src / "__pycache__"
    gitdir = base / ".git"
    src.mkdir(parents=True)
    pycache.mkdir(parents=True)
    gitdir.mkdir(parents=True)

    (src / "a.py").write_text("print('ok')\n", encoding="utf-8")
    (src / "b.pyc").write_bytes(b"x")
    (pycache / "c.py").write_text("skip\n", encoding="utf-8")
    (gitdir / "config").write_text("[core]\n", encoding="utf-8")

    out = tmp_path / "code.zip"
    zip_sources(out, base, include=["src", ".git"])

    with zipfile.ZipFile(out, "r") as zf:
        names = sorted(zf.namelist())

    assert "src/a.py" in names
    assert "src/b.pyc" not in names
    assert "src/__pycache__/c.py" not in names
    assert ".git/config" not in names


def test_zip_sources_ignores_missing_include_paths(tmp_path: Path):
    base = tmp_path / "project"
    base.mkdir(parents=True)
    out = tmp_path / "empty.zip"

    zip_sources(out, base, include=["missing_dir", "missing_file.py"])

    with zipfile.ZipFile(out, "r") as zf:
        assert zf.namelist() == []
