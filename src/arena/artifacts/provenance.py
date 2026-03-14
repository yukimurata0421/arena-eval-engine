from __future__ import annotations

import json
from pathlib import Path

from arena.artifacts.models import AIManifestRecord
from arena.artifacts.policies import AI_ARTIFACT_PROVENANCE_FILENAME, POLICY_VERSION
from arena.artifacts.schema import validate_artifact_provenance


def build_artifact_provenance(records: list[AIManifestRecord], generated_at: str) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for record in records:
        if not record.copied or not record.copied_path or not record.artifact_sha256 or not record.source_path:
            continue
        entries.append(
            {
                "artifact_path": record.copied_path,
                "artifact_sha256": record.artifact_sha256,
                "source_paths": [record.source_path],
                "generation_stage": "ai_export_copy",
                "generation_timestamp": generated_at,
                "policy_version": POLICY_VERSION,
            }
        )

    payload = {
        "generated_at": generated_at,
        "policy_version": POLICY_VERSION,
        "entries": sorted(entries, key=lambda entry: str(entry["artifact_path"])),
    }
    validate_artifact_provenance(payload)
    return payload


def write_artifact_provenance(export_dir: Path, records: list[AIManifestRecord], generated_at: str) -> Path:
    payload = build_artifact_provenance(records, generated_at)
    destination = export_dir / AI_ARTIFACT_PROVENANCE_FILENAME
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return destination

