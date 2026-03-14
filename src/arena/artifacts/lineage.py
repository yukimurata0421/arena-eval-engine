from __future__ import annotations

import json
from pathlib import Path

from arena.artifacts.models import AIManifestRecord
from arena.artifacts.policies import AI_ARTIFACT_LINEAGE_FILENAME, POLICY_VERSION
from arena.artifacts.schema import validate_artifact_lineage


def build_artifact_lineage(records: list[AIManifestRecord], generated_at: str) -> dict[str, object]:
    nodes: dict[str, dict[str, str]] = {}
    edges: list[dict[str, str]] = []
    artifact_types: dict[str, str] = {}

    for record in records:
        if record.source_path:
            source_id = f"source:{record.source_path}"
            nodes.setdefault(
                source_id,
                {
                    "id": source_id,
                    "path": record.source_path,
                    "node_kind": "source",
                    "category": "source",
                    "status": "source",
                },
            )
            artifact_types.setdefault(source_id, "source")

        if not record.copied or not record.copied_path:
            continue

        nodes.setdefault(
            record.copied_path,
            {
                "id": record.copied_path,
                "path": record.copied_path,
                "node_kind": "artifact",
                "category": record.category,
                "status": record.status,
            },
        )
        artifact_types.setdefault(record.copied_path, record.category)
        if record.source_path:
            edges.append(
                {
                    "from": f"source:{record.source_path}",
                    "to": record.copied_path,
                    "relationship": "derived_from",
                }
            )

    payload = {
        "generated_at": generated_at,
        "policy_version": POLICY_VERSION,
        "nodes": sorted(nodes.values(), key=lambda node: node["id"]),
        "edges": sorted(edges, key=lambda edge: (edge["from"], edge["to"], edge["relationship"])),
        "artifact_types": dict(sorted(artifact_types.items())),
    }
    validate_artifact_lineage(payload)
    return payload


def write_artifact_lineage(export_dir: Path, records: list[AIManifestRecord], generated_at: str) -> Path:
    payload = build_artifact_lineage(records, generated_at)
    destination = export_dir / AI_ARTIFACT_LINEAGE_FILENAME
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return destination

