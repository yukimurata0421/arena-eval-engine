from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


@dataclass
class SettingsSnapshot:
    path: str
    data: dict[str, Any]


def find_settings_path() -> Path:
    env_path = os.getenv("ARENA_SETTINGS") or os.getenv("ADSB_SETTINGS")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # Prefer repo scripts/config
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "scripts" / "config" / "settings.toml",
    ]
    try:
        from arena.lib.paths import SCRIPTS_ROOT

        candidates.insert(0, SCRIPTS_ROOT / "config" / "settings.toml")
    except Exception:
        pass
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_settings() -> SettingsSnapshot:
    path = find_settings_path()
    if tomllib is None:
        return SettingsSnapshot(path=str(path), data={"error": "tomllib not available"})
    if not path.exists():
        return SettingsSnapshot(path=str(path), data={})
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        return SettingsSnapshot(path=str(path), data={"error": f"failed_to_parse: {exc}"})
    return SettingsSnapshot(path=str(path), data=data)


def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def build_snapshot(phase_config_path: str) -> dict[str, Any]:
    settings = load_settings()
    snapshot = {
        "settings_path": settings.path,
        "settings": settings.data,
        "phase_config_path": phase_config_path,
        "phase_config_text": load_text(phase_config_path),
    }
    return snapshot
