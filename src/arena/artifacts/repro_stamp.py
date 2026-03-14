from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

from arena.artifacts.policies import (
    AI_REPRODUCIBILITY_STAMP_FILENAME,
    ARTIFACT_SUBSYSTEM_VERSION,
    POLICY_VERSION,
    ROOT,
)


def resolve_generated_at(deterministic: bool) -> str:
    if deterministic:
        return "1970-01-01T00:00:00"
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


def resolve_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def write_reproducibility_stamp(
    export_dir: Path,
    export_mode: str,
    deterministic: bool,
) -> Path:
    payload = {
        "timestamp": resolve_generated_at(deterministic),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": resolve_git_commit(),
        "artifact_subsystem_version": ARTIFACT_SUBSYSTEM_VERSION,
        "export_mode": export_mode,
        "deterministic_flag": deterministic,
        "policy_version": POLICY_VERSION,
    }
    stamp_path = export_dir / AI_REPRODUCIBILITY_STAMP_FILENAME
    with stamp_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return stamp_path

