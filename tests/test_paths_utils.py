from __future__ import annotations

from arena.lib import paths


def test_win_to_wsl_path_conversion() -> None:
    assert paths._win_to_wsl_path(r"E:\arena\data") == "/mnt/e/arena/data"


def test_wsl_to_win_path_conversion() -> None:
    assert paths._wsl_to_win_path("/mnt/e/arena/data") == r"E:\arena\data"


def test_looks_like_windows_path() -> None:
    assert paths._looks_like_windows_path(r"C:\x") is True
    assert paths._looks_like_windows_path("/mnt/c/x") is False
