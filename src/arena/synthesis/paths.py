from __future__ import annotations

import os
from pathlib import Path

# SYNTHESIS invariants (do not change silently):
# - Default variable data root is workspace/synthesis
# - SQLite DB filename is fixed to synthesis.sqlite3 (single DB target)
# - ARENA_SYNTHESIS_DIR can override the synthesis workspace root

def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Project root not found from synthesis.paths")


def _resolve_synthesis_dir(project_root: Path) -> Path:
    configured = os.getenv("ARENA_SYNTHESIS_DIR")
    if not configured:
        return (project_root / "workspace" / "synthesis").resolve()

    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
SYNTHESIS_DIR = _resolve_synthesis_dir(PROJECT_ROOT)

RAW_DIR = SYNTHESIS_DIR / "raw"
RAW_CLAUDE_DIR = RAW_DIR / "claude"
RAW_GEMINI_DIR = RAW_DIR / "gemini"
RAW_GPT_DIR = RAW_DIR / "gpt"
RAW_GROK_DIR = RAW_DIR / "grok"
RAW_ORIGINAL_DIR = SYNTHESIS_DIR / "raw_original"
RAW_REPAIRED_DIR = SYNTHESIS_DIR / "raw_repaired"
REPAIR_LOG_DIR = SYNTHESIS_DIR / "repair_logs"
ENRICHED_DIR = SYNTHESIS_DIR / "enriched"

DB_DIR = SYNTHESIS_DIR / "db"
EXPORTS_DIR = SYNTHESIS_DIR / "exports"
REVIEW_DIR = SYNTHESIS_DIR / "review"
CANONICAL_DB_FILENAME = "synthesis.sqlite3"
DB_PATH = DB_DIR / CANONICAL_DB_FILENAME
LEGACY_DB_PATH = DB_DIR / "arena_synthesis.db"


def legacy_db_paths_present() -> list[Path]:
    return [LEGACY_DB_PATH] if LEGACY_DB_PATH.exists() else []


def as_dict() -> dict[str, str]:
    return {
        "project_root": str(PROJECT_ROOT),
        "src_dir": str(SRC_DIR),
        "workspace_dir": str(WORKSPACE_DIR),
        "synthesis_dir": str(SYNTHESIS_DIR),
        "raw_dir": str(RAW_DIR),
        "raw_claude_dir": str(RAW_CLAUDE_DIR),
        "raw_gemini_dir": str(RAW_GEMINI_DIR),
        "raw_gpt_dir": str(RAW_GPT_DIR),
        "raw_grok_dir": str(RAW_GROK_DIR),
        "raw_original_dir": str(RAW_ORIGINAL_DIR),
        "raw_repaired_dir": str(RAW_REPAIRED_DIR),
        "repair_log_dir": str(REPAIR_LOG_DIR),
        "enriched_dir": str(ENRICHED_DIR),
        "db_dir": str(DB_DIR),
        "exports_dir": str(EXPORTS_DIR),
        "review_dir": str(REVIEW_DIR),
        "canonical_db_filename": CANONICAL_DB_FILENAME,
        "db_path": str(DB_PATH),
        "legacy_db_paths_present": ";".join(str(p) for p in legacy_db_paths_present()),
    }
