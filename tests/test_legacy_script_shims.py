from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from arena.lib import phase_config as lib_phase_config


def test_scripts_phase_config_shim_maps_to_library_module() -> None:
    import scripts.phase_config as shim_phase_config

    assert hasattr(shim_phase_config, "get_config")
    assert shim_phase_config.get_config is lib_phase_config.get_config


def test_scripts_master_wrapper_delegates_to_arena_cli(monkeypatch, capsys) -> None:
    import arena.cli as arena_cli

    called: dict[str, object] = {}

    def fake_main(argv):
        called["argv"] = list(argv)
        return 0

    monkeypatch.setattr(arena_cli, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["scripts/master.py", "--only", "1", "--no-gpu"])

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "master.py"
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script_path), run_name="__main__")

    assert exc.value.code == 0
    assert called.get("argv") == ["run", "--only", "1", "--no-gpu"]
    assert "deprecated" in capsys.readouterr().out.lower()

