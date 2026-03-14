from __future__ import annotations

import time
from pathlib import Path

from arena.pipeline.stages import validate_outputs


def test_validate_outputs_file_success(tmp_path: Path) -> None:
    f = tmp_path / "a.csv"
    f.write_text("x" * 100, encoding="utf-8")
    ok, missing = validate_outputs(tmp_path, ["a.csv"], min_bytes=50)
    assert ok is True
    assert missing == []


def test_validate_outputs_detects_empty_dir(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    ok, missing = validate_outputs(tmp_path, ["empty"], min_bytes=1)
    assert ok is False
    assert any("empty dir" in m for m in missing)


def test_validate_outputs_detects_stale_file(tmp_path: Path) -> None:
    f = tmp_path / "old.csv"
    f.write_text("x" * 100, encoding="utf-8")
    min_mtime = time.time() + 60
    ok, missing = validate_outputs(tmp_path, ["old.csv"], min_bytes=1, min_mtime=min_mtime)
    assert ok is False
    assert any("stale:" in m for m in missing)


def test_validate_outputs_resolves_data_logical_path(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    out = data_root / "flight_data" / "airport_movements.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("x" * 300, encoding="utf-8")

    ok, missing = validate_outputs(
        tmp_path / "output",
        ["data://flight_data/airport_movements.csv"],
        min_bytes=200,
        data_root_native=data_root,
    )
    assert ok is True
    assert missing == []
