from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from arena.lib._toml_compat import parse_settings_fallback, tomllib


def _iter_repo_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    try:
        cwd = Path.cwd()
    except Exception:
        cwd = None
    if cwd is not None:
        add(cwd)
        for parent in cwd.parents:
            add(parent)

    here = Path(__file__).resolve()
    add(here)
    for parent in here.parents:
        add(parent)
    return candidates


def find_scripts_root() -> Path:
    env_scripts = os.getenv("ARENA_SCRIPTS_ROOT") or os.getenv("ADSB_SCRIPTS_ROOT")
    if env_scripts:
        return Path(env_scripts)

    # Search the active checkout first so regular installs from a repo root
    # behave the same as editable installs during CI and local smoke runs.
    for parent in _iter_repo_root_candidates():
        candidate = parent / "scripts"
        if candidate.exists() and (candidate / "adsb").exists():
            return candidate
        if parent.name == "scripts" and (parent / "adsb").exists():
            return parent
    return Path(__file__).resolve().parents[1]


def find_settings_path() -> Path:
    env_path = os.getenv("ARENA_SETTINGS") or os.getenv("ADSB_SETTINGS")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    scripts_root = find_scripts_root()
    candidates.append(scripts_root / "config" / "settings.toml")
    for parent in _iter_repo_root_candidates():
        candidates.append(parent / "scripts" / "config" / "settings.toml")
        candidates.append(parent / "config" / "settings.toml")

    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_settings_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}

    if tomllib is None:
        try:
            data = parse_settings_fallback(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    try:
        data = tomllib.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover
        # Keep partial operability even when TOML is malformed.
        try:
            fallback_data = parse_settings_fallback(text)
            if isinstance(fallback_data, dict) and fallback_data:
                fallback_data["parse_warning"] = f"failed_to_parse_toml: {exc}"
                return fallback_data
        except Exception:
            pass
        return {"error": f"failed_to_parse: {exc}"}
