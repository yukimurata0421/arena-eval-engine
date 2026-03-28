from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import arena.synthesis.paths as synthesis_paths


def _reload_paths() -> object:
    return importlib.reload(synthesis_paths)


@pytest.fixture(autouse=True)
def _restore_default_paths(monkeypatch):
    yield
    monkeypatch.delenv("ARENA_SYNTHESIS_DIR", raising=False)
    _reload_paths()


def test_paths_default_synthesis_dir_under_workspace(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SYNTHESIS_DIR", raising=False)
    paths = _reload_paths()
    assert Path(paths.SYNTHESIS_DIR) == Path(paths.PROJECT_ROOT) / "workspace" / "synthesis"
    assert Path(paths.DB_PATH) == Path(paths.SYNTHESIS_DIR) / "db" / "synthesis.sqlite3"


def test_paths_support_absolute_synthesis_dir_override(monkeypatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom_synthesis"
    monkeypatch.setenv("ARENA_SYNTHESIS_DIR", str(custom))
    paths = _reload_paths()
    assert Path(paths.SYNTHESIS_DIR) == custom.resolve()
    assert Path(paths.RAW_DIR) == custom.resolve() / "raw"
    assert Path(paths.DB_PATH) == custom.resolve() / "db" / "synthesis.sqlite3"


def test_paths_support_relative_synthesis_dir_override(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_SYNTHESIS_DIR", "tmp/synthesis_rel")
    paths = _reload_paths()
    expected = (Path(paths.PROJECT_ROOT) / "tmp" / "synthesis_rel").resolve()
    assert Path(paths.SYNTHESIS_DIR) == expected
    assert Path(paths.ENRICHED_DIR) == expected / "enriched"
