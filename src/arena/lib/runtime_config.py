from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arena.lib.config_resolution import build_runtime_config_metadata
from arena.lib.settings_loader import find_settings_path as _find_settings_path
from arena.lib.settings_loader import load_settings_data


@dataclass
class SettingsSnapshot:
    path: str
    data: dict[str, Any]


_settings_cache: SettingsSnapshot | None = None


def find_settings_path() -> Path:
    return _find_settings_path()


def load_settings(force_reload: bool = False) -> SettingsSnapshot:
    global _settings_cache
    path = find_settings_path()
    path_str = str(path)
    if (not force_reload) and _settings_cache is not None and _settings_cache.path == path_str:
        return _settings_cache

    data = load_settings_data(path)
    snapshot = SettingsSnapshot(path=path_str, data=data if isinstance(data, dict) else {})
    _settings_cache = snapshot
    return snapshot


def clear_settings_cache() -> None:
    global _settings_cache
    _settings_cache = None


def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def build_snapshot(
    phase_config_path: str,
    *,
    settings_path: str | None = None,
    scripts_root: str | None = None,
    resolution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    metadata = resolution_metadata or build_runtime_config_metadata(
        settings_override=settings_path,
        phase_override=phase_config_path,
        scripts_root=scripts_root,
        analysis_start_date=os.getenv("ARENA_ANALYSIS_START_DATE", ""),
        analysis_end_date=os.getenv("ARENA_ANALYSIS_END_DATE", ""),
    )
    snapshot = {
        "settings_path": settings.path,
        "settings": settings.data,
        "phase_config_path": phase_config_path,
        "phase_config_text": load_text(phase_config_path),
        "resolved_settings_path": metadata["resolved_settings_path"],
        "resolved_phase_config_path": metadata["resolved_phase_config_path"],
        "used_default_settings": metadata["used_default_settings"],
        "used_default_phase_config": metadata["used_default_phase_config"],
        "analysis_start_date": metadata["analysis_start_date"],
        "analysis_end_date": metadata["analysis_end_date"],
        "experimental_mode": metadata["experimental_mode"],
        "settings_resolution_source": metadata["settings_resolution_source"],
        "phase_resolution_source": metadata["phase_resolution_source"],
    }
    return snapshot
