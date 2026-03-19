from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from arena.lib.settings_loader import find_scripts_root

MAIN_PHASE_CONFIG_NAME = "phases.txt"
MAIN_SETTINGS_NAME = "settings.toml"
EXPERIMENTAL_PHASE_CONFIG_NAME = "phases_v3_airspy_baseline.txt"
EXPERIMENTAL_SETTINGS_NAME = "settings_experimental_distance_bins.toml"


def _norm_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def default_settings_path(scripts_root: str | Path | None = None) -> Path:
    root = _norm_path(scripts_root) if scripts_root else find_scripts_root().resolve()
    return root / "config" / MAIN_SETTINGS_NAME


def default_phase_config_path(scripts_root: str | Path | None = None) -> Path:
    root = _norm_path(scripts_root) if scripts_root else find_scripts_root().resolve()
    return root / "config" / MAIN_PHASE_CONFIG_NAME


def resolve_settings_path(
    *,
    settings_override: str | Path | None = None,
    scripts_root: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[Path, bool, str]:
    active_env = env if env is not None else os.environ
    default_path = default_settings_path(scripts_root=scripts_root)
    if settings_override:
        return _norm_path(settings_override), False, "cli"
    env_override = active_env.get("ARENA_SETTINGS") or active_env.get("ADSB_SETTINGS")
    if env_override:
        return _norm_path(env_override), False, "env"
    return default_path, True, "default"


def resolve_phase_config_path(
    *,
    phase_override: str | Path | None = None,
    scripts_root: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[Path, bool, str]:
    active_env = env if env is not None else os.environ
    default_path = default_phase_config_path(scripts_root=scripts_root)
    if phase_override:
        return _norm_path(phase_override), False, "cli"
    env_override = active_env.get("ARENA_PHASE_CONFIG") or active_env.get("ADSB_PHASE_CONFIG")
    if env_override:
        return _norm_path(env_override), False, "env"
    return default_path, True, "default"


def _is_experimental_path(path: Path) -> bool:
    name = path.name.strip().lower()
    return name in {
        EXPERIMENTAL_PHASE_CONFIG_NAME.lower(),
        EXPERIMENTAL_SETTINGS_NAME.lower(),
    }


def build_runtime_config_metadata(
    *,
    settings_override: str | Path | None = None,
    phase_override: str | Path | None = None,
    scripts_root: str | Path | None = None,
    analysis_start_date: str | None = None,
    analysis_end_date: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    resolved_settings, used_default_settings, settings_source = resolve_settings_path(
        settings_override=settings_override,
        scripts_root=scripts_root,
        env=env,
    )
    resolved_phase, used_default_phase, phase_source = resolve_phase_config_path(
        phase_override=phase_override,
        scripts_root=scripts_root,
        env=env,
    )
    default_settings = default_settings_path(scripts_root=scripts_root)
    default_phase = default_phase_config_path(scripts_root=scripts_root)
    used_default_settings = resolved_settings == default_settings
    used_default_phase = resolved_phase == default_phase

    start_date = (analysis_start_date or "").strip()
    end_date = (analysis_end_date or "").strip()
    experimental_mode = _is_experimental_path(resolved_settings) or _is_experimental_path(resolved_phase)

    return {
        "resolved_settings_path": str(resolved_settings),
        "resolved_phase_config_path": str(resolved_phase),
        "default_settings_path": str(default_settings),
        "default_phase_config_path": str(default_phase),
        "used_default_settings": bool(used_default_settings),
        "used_default_phase_config": bool(used_default_phase),
        "settings_resolution_source": settings_source,
        "phase_resolution_source": phase_source,
        "analysis_start_date": start_date,
        "analysis_end_date": end_date,
        "experimental_mode": bool(experimental_mode),
    }


def validate_resolved_config_paths(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    settings_path = Path(str(metadata.get("resolved_settings_path", "")))
    phase_path = Path(str(metadata.get("resolved_phase_config_path", "")))

    if not settings_path.exists():
        errors.append(
            "settings file not found: {path} (default={default}, source={source})".format(
                path=settings_path,
                default=metadata.get("default_settings_path", ""),
                source=metadata.get("settings_resolution_source", ""),
            )
        )
    if not phase_path.exists():
        errors.append(
            "phase config file not found: {path} (default={default}, source={source})".format(
                path=phase_path,
                default=metadata.get("default_phase_config_path", ""),
                source=metadata.get("phase_resolution_source", ""),
            )
        )
    return errors
