from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from arena.lib.runtime_config import load_settings
from arena.lib.settings_loader import find_scripts_root


def _looks_like_windows_path(val: str) -> bool:
    if len(val) < 2:
        return False
    return val[1] == ":" and (len(val) == 2 or val[2] in ("\\", "/"))


def _win_to_wsl_path(val: str) -> str:
    drive = val[0].lower()
    rest = val[2:].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{rest}"


def _wsl_to_win_path(val: str) -> str:
    # /mnt/e/foo/bar -> E:\foo\bar
    parts = val.split("/", 3)
    if len(parts) < 3:
        return val
    drive = parts[2].upper()
    rest = parts[3] if len(parts) > 3 else ""
    rest = rest.replace("/", "\\")
    return f"{drive}:\\" + rest


def _load_settings() -> dict[str, Any]:
    data = load_settings().data
    return data if isinstance(data, dict) else {}


def _settings_path_value(key: str, settings: dict[str, Any] | None = None) -> Path | None:
    source = settings if settings is not None else _load_settings()
    paths = source.get("paths", {})
    if not isinstance(paths, dict):
        return None
    val = paths.get(key)
    if val is None:
        return None
    raw = str(val).strip()
    if not raw or raw.lower() in {"auto", "none"}:
        return None
    if os.name != "nt" and _looks_like_windows_path(raw):
        raw = _win_to_wsl_path(raw)
    elif os.name == "nt" and raw.startswith("/mnt/") and len(raw) > 6:
        raw = _wsl_to_win_path(raw)
    return Path(raw)


def resolve_scripts_root() -> Path:
    return find_scripts_root()


def resolve_root(
    *,
    settings: dict[str, Any] | None = None,
    scripts_root: Path | None = None,
) -> Path:
    env_root = os.getenv("ARENA_ROOT") or os.getenv("ADSB_ROOT")
    if env_root:
        return Path(env_root)
    settings_root = _settings_path_value("root", settings=settings)
    if settings_root:
        return settings_root

    # Prefer project root inferred from scripts location
    scripts = scripts_root or resolve_scripts_root()
    try:
        if scripts.exists() and scripts.parent.exists():
            return scripts.parent
    except Exception:
        pass

    # Fallback: current working directory (repo root expected)
    try:
        return Path.cwd()
    except Exception:
        return Path("/")

def resolve_data_dir(*, root: Path | None = None, settings: dict[str, Any] | None = None) -> Path:
    env_data = os.getenv("ARENA_DATA_DIR") or os.getenv("ADSB_DATA_DIR")
    if env_data:
        return Path(env_data)
    resolved_root = root or resolve_root(settings=settings)
    return _settings_path_value("data_dir", settings=settings) or (resolved_root / "data")


def resolve_output_dir(*, root: Path | None = None, settings: dict[str, Any] | None = None) -> Path:
    env_output = os.getenv("ARENA_OUTPUT_DIR") or os.getenv("ADSB_OUTPUT_DIR")
    if env_output:
        return Path(env_output)
    resolved_root = root or resolve_root(settings=settings)
    return _settings_path_value("output_dir", settings=settings) or (resolved_root / "output")


def resolve_runtime_roots() -> tuple[Path, Path, Path]:
    scripts_root = resolve_scripts_root()
    settings = _load_settings()
    root = resolve_root(settings=settings, scripts_root=scripts_root)
    output = resolve_output_dir(root=root, settings=settings)
    data = resolve_data_dir(root=root, settings=settings)
    return scripts_root, output, data


# Backward-compatible module constants (snapshot at import time).
SCRIPTS_ROOT, OUTPUT_DIR, DATA_DIR = resolve_runtime_roots()
ROOT = resolve_root(scripts_root=SCRIPTS_ROOT)
RAW_DIR = DATA_DIR / "raw"
PAST_LOG_DIR = RAW_DIR / "past_log"

ADSB_DAILY_SUMMARY = OUTPUT_DIR / "adsb_daily_summary.csv"
ADSB_DAILY_SUMMARY_V2 = OUTPUT_DIR / "adsb_daily_summary_v2.csv"
ADSB_DAILY_SUMMARY_RAW = OUTPUT_DIR / "adsb_daily_summary_raw.csv"
ADSB_SIGNAL_RANGE_SUMMARY = OUTPUT_DIR / "adsb_signal_range_summary.csv"
ADSB_SIGNAL_DAILY_SUMMARY = OUTPUT_DIR / "adsb_signal_daily_summary.csv"


def ensure_dir(path: Path) -> None:
    os.makedirs(path, exist_ok=True)
