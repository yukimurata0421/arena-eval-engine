from __future__ import annotations

import hashlib
import json
from pathlib import Path

from arena.artifacts.models import AIManifestRecord


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_record_hashes(export_dir: Path, records: list[AIManifestRecord]) -> None:
    for record in records:
        if not record.copied or not record.copied_path:
            record.artifact_sha256 = ""
            continue
        exported_path = export_dir / record.copied_path
        if exported_path.exists() and exported_path.is_file():
            record.artifact_sha256 = sha256_file(exported_path)
        else:
            record.artifact_sha256 = ""


def write_artifact_hashes(hash_path: Path, records: list[AIManifestRecord]) -> Path:
    lines: list[str] = []
    for record in records:
        if not record.copied or not record.copied_path or not record.artifact_sha256:
            continue
        lines.append(f"{record.artifact_sha256}  {record.copied_path}")

    hash_path.parent.mkdir(parents=True, exist_ok=True)
    with hash_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + ("\n" if lines else ""))
    return hash_path


def read_artifact_hashes(hash_path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not hash_path.exists():
        return hashes
    with hash_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            digest, relative_path = stripped.split("  ", 1)
            hashes[relative_path] = digest
    return hashes


def stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_bundle_sha256(
    artifact_hashes: dict[str, str],
    manifest_rows: list[dict[str, object]],
    schema_payload: dict[str, object],
) -> str:
    canonical_payload = {
        "artifact_hashes": [[relative_path, artifact_hashes[relative_path]] for relative_path in sorted(artifact_hashes)],
        "manifest_rows": manifest_rows,
        "schema": schema_payload,
    }
    return hashlib.sha256(stable_json_dumps(canonical_payload).encode("utf-8")).hexdigest()

