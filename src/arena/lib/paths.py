import os
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


def _find_scripts_root() -> Path:
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


SCRIPTS_ROOT = _find_scripts_root()


def _load_settings() -> dict:
    env_path = os.getenv("ARENA_SETTINGS") or os.getenv("ADSB_SETTINGS")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(SCRIPTS_ROOT / "config" / "settings.toml")
    # src layout fallback
    candidates.append(Path(__file__).resolve().parents[2] / "config" / "settings.toml")
    for p in candidates:
        if p.exists() and tomllib is not None:
            try:
                return tomllib.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


_SETTINGS = _load_settings()


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


def _settings_path_value(key: str) -> Path | None:
    val = _SETTINGS.get("paths", {}).get(key)
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


def _resolve_root() -> Path:
    env_root = os.getenv("ARENA_ROOT") or os.getenv("ADSB_ROOT")
    if env_root:
        return Path(env_root)
    settings_root = _settings_path_value("root")
    if settings_root:
        return settings_root

    # Prefer project root inferred from scripts location
    try:
        if SCRIPTS_ROOT.exists() and SCRIPTS_ROOT.parent.exists():
            return SCRIPTS_ROOT.parent
    except Exception:
        pass

    # Fallback: current working directory (repo root expected)
    try:
        return Path.cwd()
    except Exception:
        return Path("/")

ROOT = _resolve_root()
_ENV_DATA = os.getenv("ARENA_DATA_DIR") or os.getenv("ADSB_DATA_DIR")
_ENV_OUTPUT = os.getenv("ARENA_OUTPUT_DIR") or os.getenv("ADSB_OUTPUT_DIR")

DATA_DIR = Path(_ENV_DATA) if _ENV_DATA else (_settings_path_value("data_dir") or (ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PAST_LOG_DIR = RAW_DIR / "past_log"
OUTPUT_DIR = Path(_ENV_OUTPUT) if _ENV_OUTPUT else (_settings_path_value("output_dir") or (ROOT / "output"))

ADSB_DAILY_SUMMARY = OUTPUT_DIR / "adsb_daily_summary.csv"
ADSB_DAILY_SUMMARY_V2 = OUTPUT_DIR / "adsb_daily_summary_v2.csv"
ADSB_DAILY_SUMMARY_RAW = OUTPUT_DIR / "adsb_daily_summary_raw.csv"
ADSB_SIGNAL_RANGE_SUMMARY = OUTPUT_DIR / "adsb_signal_range_summary.csv"
ADSB_SIGNAL_DAILY_SUMMARY = OUTPUT_DIR / "adsb_signal_daily_summary.csv"


def ensure_dir(path: Path):
    os.makedirs(path, exist_ok=True)
