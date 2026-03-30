from pathlib import Path

from chimera_ml.logging.utils import generate_run_name, local_datetime_tag, short_hash


def test_local_datetime_tag_formats():
    d = local_datetime_tag(include_time=False)
    assert len(d) == 10
    assert d.count("-") == 2

    dt = local_datetime_tag(include_time=True, fmt="%Y%m%d")
    assert len(dt) == 8
    assert dt.isdigit()


def test_short_hash_length():
    h = short_hash("abc", n=6)
    assert len(h) == 6
    assert h == short_hash("abc", n=6)


def test_generate_run_name_includes_config_stem_model_and_hash(tmp_path: Path):
    cfg = tmp_path / "demo_cfg.yaml"
    cfg.write_text("seed: 1\n", encoding="utf-8")
    name = generate_run_name(
        config_path=str(cfg),
        model_name="m1",
        suffix="x",
        include_time=False,
        datetime_format="%Y%m%d",
    )
    assert "demo_cfg" in name
    assert "m1" in name
    assert "x" in name
