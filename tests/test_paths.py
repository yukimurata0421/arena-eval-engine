from pathlib import Path

from arena.lib import paths


def test_ensure_dir(tmp_path: Path):
    target = tmp_path / "nested" / "dir"
    paths.ensure_dir(target)
    assert target.exists() and target.is_dir()


def test_wsl_win_roundtrip():
    win = "E:\\arena_public\\data"
    wsl = paths._win_to_wsl_path(win)  # type: ignore[attr-defined]
    back = paths._wsl_to_win_path(wsl)  # type: ignore[attr-defined]
    assert back.startswith("E:\\")
