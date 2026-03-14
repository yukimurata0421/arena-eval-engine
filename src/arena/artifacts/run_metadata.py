from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

from arena.artifacts.models import AIManifestRecord
from arena.artifacts.policies import AI_RUN_METADATA_FILENAME
from arena.artifacts.repro_stamp import resolve_git_commit
from arena.artifacts.schema import validate_run_metadata


def build_run_metadata(
    export_dir: Path,
    records: list[AIManifestRecord],
    generated_at: str,
    deterministic: bool,
) -> dict[str, object]:
    payload = {
        "run_id": "deterministic-ai-export" if deterministic else export_dir.name,
        "timestamp": generated_at,
        "hostname": platform.node(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": resolve_git_commit(),
        "deterministic_flag": deterministic,
        "artifact_count": sum(1 for record in records if record.copied),
        "missing_required_count": sum(1 for record in records if record.status == "missing_required"),
        "missing_recommended_count": sum(1 for record in records if record.status == "missing_recommended"),
        "excluded_count": sum(1 for record in records if record.status == "excluded_by_rule"),
    }
    validate_run_metadata(payload)
    return payload


def write_run_metadata(
    export_dir: Path,
    records: list[AIManifestRecord],
    generated_at: str,
    deterministic: bool,
) -> Path:
    payload = build_run_metadata(export_dir, records, generated_at, deterministic)
    destination = export_dir / AI_RUN_METADATA_FILENAME
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return destination

