from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from arena.lib._toml_compat import parse_settings_fallback, tomllib


def find_scripts_root() -> Path:
    env_scripts = os.getenv("ARENA_SCRIPTS_ROOT") or os.getenv("ADSB_SCRIPTS_ROOT")
    if env_scripts:
        return Path(env_scripts)

    here = Path(__file__).resolve()
    # If running from src layout, find the sibling "scripts" directory.
    for parent in here.parents:
        candidate = parent / "scripts"
        if candidate.exists() and (candidate / "adsb").exists():
            return candidate
        if parent.name == "scripts" and (parent / "adsb").exists():
            return parent
    return here.parents[1]


def find_settings_path() -> Path:
    env_path = os.getenv("ARENA_SETTINGS") or os.getenv("ADSB_SETTINGS")
    if env_path:
        return Path(env_path)

    scripts_root = find_scripts_root()
    return scripts_root / "config" / "settings.toml"


def load_settings_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(
            f"[WARN] settings.toml の読み込みに失敗しました ({path}): " f"{type(e).__name__}: {e} - デフォルト設定を使用します",
            file=sys.stderr,
        )
        return {}

    if tomllib is None:
        try:
            data = parse_settings_fallback(text)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(
                f"[WARN] settings.toml のパースに失敗しました ({path}): " f"{type(e).__name__}: {e} - デフォルト設定を使用します",
                file=sys.stderr,
            )
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
                print(
                    f"[WARN] settings.toml のTOMLパースに失敗、フォールバックパーサを使用: {exc}",
                    file=sys.stderr,
                )
                return fallback_data
        except Exception as e2:
            print(
                f"[WARN] settings.toml のパースが完全に失敗しました ({path}): {exc}; fallback: {e2}" " - デフォルト設定を使用します",
                file=sys.stderr,
            )
        return {"error": f"failed_to_parse: {exc}"}
