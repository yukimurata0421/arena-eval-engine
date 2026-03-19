from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _ensure_paths_on_sys_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_paths_on_sys_path()


def _guard_cli_env(monkeypatch) -> None:
    leaked = (
        "ARENA_SCRIPTS_ROOT",
        "ARENA_DATA_DIR",
        "ARENA_OUTPUT_DIR",
        "ARENA_SETTINGS",
        "ARENA_PHASE_CONFIG",
        "ADSB_SETTINGS",
        "ADSB_PHASE_CONFIG",
        "ARENA_ANALYSIS_START_DATE",
        "ARENA_ANALYSIS_END_DATE",
    )
    for k in leaked:
        monkeypatch.setenv(k, "")
    for k in leaked:
        monkeypatch.delenv(k)


@pytest.fixture
def cli_env(monkeypatch, tmp_path: Path):
    """Standard CLI test environment (dirs, settings/phases, env guard)."""
    _guard_cli_env(monkeypatch)
    scripts_root = tmp_path / "scripts"
    (scripts_root / "adsb").mkdir(parents=True)
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()

    settings_path = tmp_path / "settings.toml"
    settings_path.write_text("x\n", encoding="utf-8", newline="\n")
    phase_path = tmp_path / "phases.txt"
    phase_path.write_text("x\n", encoding="utf-8", newline="\n")

    monkeypatch.setenv("ARENA_SCRIPTS_ROOT", str(scripts_root))
    monkeypatch.setenv("ARENA_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ARENA_OUTPUT_DIR", str(output_dir))

    return SimpleNamespace(
        tmp_path=tmp_path,
        scripts_root=scripts_root,
        data_dir=data_dir,
        output_dir=output_dir,
        settings_path=settings_path,
        phase_path=phase_path,
    )
