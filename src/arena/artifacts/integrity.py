from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

from arena.artifacts.hash_utils import compute_bundle_sha256, read_artifact_hashes, sha256_file
from arena.artifacts.models import AIExportIntegrity, AIManifestRecord
from arena.artifacts.policies import (
    AI_ARTIFACT_HASHES_FILENAME,
    AI_ARTIFACT_INDEX_FILENAME,
    AI_ARTIFACT_LINEAGE_FILENAME,
    AI_ARTIFACT_PROVENANCE_FILENAME,
    AI_CANDIDATE_STATUS_CSV_FILENAME,
    AI_INTEGRITY_SUMMARY_JSON_FILENAME,
    AI_MANIFEST_EXTENDED_FILENAME,
    AI_MANIFEST_FILENAME,
    AI_PACK_DIR_GEMINI,
    AI_PACK_DIR_GPT,
    AI_PACK_DIR_GROK,
    AI_PACK_MANIFESTS_DIR,
    AI_REPRODUCIBILITY_STAMP_FILENAME,
    AI_RUN_METADATA_FILENAME,
)
from arena.artifacts.schema import (
    export_schema_catalog,
    validate_artifact_index,
    validate_artifact_lineage,
    validate_artifact_provenance,
    validate_candidate_status_rows,
    validate_integrity_summary,
    validate_manifest_record_rows,
    validate_pack_manifest_payload,
    validate_run_metadata,
)


def integrity_to_payload(integrity: AIExportIntegrity) -> dict[str, object]:
    payload: dict[str, object] = {
        "duplicate_source_skipped": integrity.duplicate_source_skipped,
        "duplicate_output_paths": integrity.duplicate_output_paths,
        "copied_records": integrity.copied_records,
        "missing_artifacts": integrity.missing_artifacts,
        "hash_mismatches": integrity.hash_mismatches,
        "validated_hash_entries": integrity.validated_hash_entries,
        "passed": integrity.passed,
    }
    validate_integrity_summary(payload)
    return payload


def run_ai_export_integrity_check(
    records: list[AIManifestRecord],
    export_dir: Path | None = None,
    hash_manifest_path: Path | None = None,
) -> AIExportIntegrity:
    copied_records = [record for record in records if record.copied]
    copied_paths = [record.copied_path for record in copied_records if record.copied_path]
    unique_paths = set(copied_paths)
    duplicate_output_paths = len(copied_paths) - len(unique_paths)
    duplicate_source_skipped = sum(1 for record in records if record.status == "duplicate_source_skipped")
    missing_artifacts = 0
    hash_mismatches = 0
    validated_hash_entries = 0

    if export_dir is not None:
        for record in copied_records:
            if not record.copied_path:
                continue
            exported_path = export_dir / record.copied_path
            if not exported_path.exists():
                missing_artifacts += 1
                continue
            if record.artifact_sha256 and sha256_file(exported_path) != record.artifact_sha256:
                hash_mismatches += 1

    if export_dir is not None and hash_manifest_path is not None and hash_manifest_path.exists():
        for relative_path, expected_hash in read_artifact_hashes(hash_manifest_path).items():
            validated_hash_entries += 1
            exported_path = export_dir / relative_path
            if not exported_path.exists():
                missing_artifacts += 1
                continue
            if sha256_file(exported_path) != expected_hash:
                hash_mismatches += 1

    passed = duplicate_output_paths == 0 and missing_artifacts == 0 and hash_mismatches == 0
    payload: dict[str, object] = {
        "duplicate_source_skipped": duplicate_source_skipped,
        "duplicate_output_paths": duplicate_output_paths,
        "copied_records": len(copied_records),
        "missing_artifacts": missing_artifacts,
        "hash_mismatches": hash_mismatches,
        "validated_hash_entries": validated_hash_entries,
        "passed": passed,
    }
    validate_integrity_summary(payload)

    return AIExportIntegrity(
        duplicate_source_skipped=duplicate_source_skipped,
        duplicate_output_paths=duplicate_output_paths,
        copied_records=len(copied_records),
        missing_artifacts=missing_artifacts,
        hash_mismatches=hash_mismatches,
        validated_hash_entries=validated_hash_entries,
        passed=passed,
    )


def write_integrity_summary_json(export_dir: Path, integrity: AIExportIntegrity) -> Path:
    payload = integrity_to_payload(integrity)
    destination = export_dir / AI_INTEGRITY_SUMMARY_JSON_FILENAME
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return destination


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _coerce_manifest_row_types(row: dict[str, str]) -> dict[str, object]:
    coerced: dict[str, object] = dict(row)
    for key in ("exists", "copied", "size_bytes"):
        coerced[key] = int(row.get(key, "0") or 0)
    return coerced


def _coerce_candidate_status_row_types(row: dict[str, str]) -> dict[str, object]:
    coerced: dict[str, object] = dict(row)
    for key in ("exists", "copied_to_export"):
        coerced[key] = int(row.get(key, "0") or 0)
    return coerced


def _manifest_rows_to_records(rows: list[dict[str, str]]) -> list[AIManifestRecord]:
    return [
        AIManifestRecord(
            relative_path=row["relative_path"],
            category=row["category"],
            required_level=row["required_level"],
            exists=bool(int(row["exists"])),
            copied=bool(int(row["copied"])),
            size_bytes=int(row["size_bytes"]),
            mtime=row["mtime"],
            status=row["status"],
            note=row["note"],
            source_path=_extract_note_field(row["note"], "source="),
            copied_path=_extract_note_field(row["note"], "copied_as="),
            artifact_sha256=row.get("artifact_sha256", ""),
        )
        for row in rows
    ]


def _extract_note_field(note: str, prefix: str) -> str:
    for part in note.split(";"):
        stripped = part.strip()
        if stripped.startswith(prefix):
            value = stripped[len(prefix):]
            return value.split(",", 1)[0].strip()
    for part in note.split(","):
        stripped = part.strip()
        if stripped.startswith(prefix):
            value = stripped[len(prefix):]
            return value.split(";", 1)[0].strip()
    return ""


def _parse_pack_manifest(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {}
    core_file_names: list[str] = []
    detail_file_examples: list[str] = []
    current_list: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":") and line in {"core_file_names:", "detail_file_examples:"}:
            current_list = line[:-1]
            continue
        if line.startswith("- "):
            if current_list == "core_file_names":
                core_file_names.append(line[2:])
            elif current_list == "detail_file_examples":
                detail_file_examples.append(line[2:])
            continue
        current_list = None
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()

    parsed = {
        "profile": payload.get("profile", ""),
        "generated_at": payload.get("generated_at", ""),
        "core_files": int(str(payload.get("core_files", "0"))),
        "detail_files": int(str(payload.get("detail_files", "0"))),
        "core_file_names": core_file_names,
        "detail_file_examples": detail_file_examples or ["(none)"],
    }
    validate_pack_manifest_payload(parsed)
    return parsed


def _resolve_pack_manifest_path(export_dir: Path, profile_dir: str) -> Path:
    new_names = {
        AI_PACK_DIR_GEMINI: "gemini_pack.txt",
        AI_PACK_DIR_GPT: "GPT_pack.txt",
        AI_PACK_DIR_GROK: "grok_pack.txt",
    }
    legacy_path = export_dir / profile_dir / "pack_manifest.txt"
    modern_path = export_dir / AI_PACK_MANIFESTS_DIR / new_names[profile_dir]
    if modern_path.exists():
        return modern_path
    return legacy_path


def _resolve_bundle_roots(bundle_path: Path) -> tuple[Path, Path]:
    """Return (metadata_dir, pack_manifest_root) for pruned or unpruned bundles.

    Unpruned bundles have metadata files at the root.  Pruned bundles
    (created with ``zip_only_profiles=True``) move all metadata into the
    ``for_grok`` pack directory while ``manifests/`` stays at root.
    """
    if (bundle_path / AI_MANIFEST_FILENAME).exists():
        return bundle_path, bundle_path
    grok_dir = bundle_path / AI_PACK_DIR_GROK
    if (grok_dir / AI_MANIFEST_FILENAME).exists():
        return grok_dir, bundle_path
    raise FileNotFoundError(
        f"Cannot locate {AI_MANIFEST_FILENAME} in {bundle_path} "
        f"or {grok_dir}"
    )


def _validate_bundle_schemas(
    metadata_dir: Path,
    pack_manifest_root: Path,
) -> dict[str, object]:
    manifest_rows = _read_csv_rows(metadata_dir / AI_MANIFEST_FILENAME)
    validate_manifest_record_rows([_coerce_manifest_row_types(row) for row in manifest_rows])

    extended_rows = _read_csv_rows(metadata_dir / AI_MANIFEST_EXTENDED_FILENAME)
    validate_manifest_record_rows([_coerce_manifest_row_types(row) for row in extended_rows])

    candidate_rows = _read_csv_rows(metadata_dir / AI_CANDIDATE_STATUS_CSV_FILENAME)
    validate_candidate_status_rows([_coerce_candidate_status_row_types(row) for row in candidate_rows])

    integrity_summary = _read_json(metadata_dir / AI_INTEGRITY_SUMMARY_JSON_FILENAME)
    validate_integrity_summary(integrity_summary)

    provenance = _read_json(metadata_dir / AI_ARTIFACT_PROVENANCE_FILENAME)
    validate_artifact_provenance(provenance)

    run_metadata = _read_json(metadata_dir / AI_RUN_METADATA_FILENAME)
    validate_run_metadata(run_metadata)

    lineage = _read_json(metadata_dir / AI_ARTIFACT_LINEAGE_FILENAME)
    validate_artifact_lineage(lineage)

    artifact_index = _read_json(metadata_dir / AI_ARTIFACT_INDEX_FILENAME)
    validate_artifact_index(artifact_index)

    pack_manifests = {
        AI_PACK_DIR_GEMINI: _parse_pack_manifest(_resolve_pack_manifest_path(pack_manifest_root, AI_PACK_DIR_GEMINI)),
        AI_PACK_DIR_GPT: _parse_pack_manifest(_resolve_pack_manifest_path(pack_manifest_root, AI_PACK_DIR_GPT)),
        AI_PACK_DIR_GROK: _parse_pack_manifest(_resolve_pack_manifest_path(pack_manifest_root, AI_PACK_DIR_GROK)),
    }
    reproducibility_stamp = _read_json(metadata_dir / AI_REPRODUCIBILITY_STAMP_FILENAME)

    return {
        "manifest_rows": manifest_rows,
        "extended_rows": extended_rows,
        "candidate_rows": candidate_rows,
        "integrity_summary": integrity_summary,
        "provenance": provenance,
        "run_metadata": run_metadata,
        "lineage": lineage,
        "artifact_index": artifact_index,
        "pack_manifests": pack_manifests,
        "reproducibility_stamp": reproducibility_stamp,
    }


def _verify_provenance_consistency(
    export_dir: Path,
    records: list[AIManifestRecord],
    artifact_hashes: dict[str, str],
    provenance: dict[str, object],
) -> list[str]:
    errors: list[str] = []
    copied_records = {record.copied_path: record for record in records if record.copied and record.copied_path}
    provenance_entries = provenance.get("entries", [])
    if not isinstance(provenance_entries, list):
        return ["artifact_provenance.entries is not a list"]

    seen_artifact_paths: set[str] = set()
    for entry in provenance_entries:
        if not isinstance(entry, dict):
            errors.append("artifact_provenance contains non-object entry")
            continue
        artifact_path = str(entry.get("artifact_path", ""))
        artifact_sha256 = str(entry.get("artifact_sha256", ""))
        source_paths = entry.get("source_paths", [])
        seen_artifact_paths.add(artifact_path)

        record = copied_records.get(artifact_path)
        if record is None:
            errors.append(f"artifact_provenance references unknown artifact_path: {artifact_path}")
            continue
        if artifact_hashes.get(artifact_path) != artifact_sha256:
            errors.append(f"artifact_provenance hash mismatch: {artifact_path}")
        if record.artifact_sha256 != artifact_sha256:
            errors.append(f"artifact_provenance disagrees with manifest hash: {artifact_path}")
        if not isinstance(source_paths, list) or record.source_path not in source_paths:
            errors.append(f"artifact_provenance missing manifest source path: {artifact_path}")
        artifact_file = export_dir / artifact_path
        if not artifact_file.exists():
            errors.append(f"artifact_provenance artifact missing on disk: {artifact_path}")

    missing_entries = sorted(path for path in copied_records if path not in seen_artifact_paths)
    for artifact_path in missing_entries:
        errors.append(f"artifact_provenance missing copied artifact: {artifact_path}")
    return errors


def verify_artifact_bundle(bundle_path: Path) -> dict[str, object]:
    export_dir = bundle_path.resolve()
    metadata_dir, pack_manifest_root = _resolve_bundle_roots(export_dir)
    schema_payload = _validate_bundle_schemas(metadata_dir, pack_manifest_root)
    manifest_rows = cast(list[dict[str, str]], schema_payload["manifest_rows"])
    records = _manifest_rows_to_records(manifest_rows)
    hash_manifest_path = metadata_dir / AI_ARTIFACT_HASHES_FILENAME
    artifact_hashes = read_artifact_hashes(hash_manifest_path)
    integrity = run_ai_export_integrity_check(
        records,
        export_dir=metadata_dir,
        hash_manifest_path=hash_manifest_path,
    )
    computed_integrity = integrity_to_payload(integrity)

    errors: list[str] = []
    if not hash_manifest_path.exists():
        errors.append("artifact_hashes.txt is missing")
    copied_record_count = sum(1 for record in records if record.copied and record.copied_path)
    if copied_record_count != len(artifact_hashes):
        errors.append("artifact_hashes.txt entry count does not match copied manifest records")
    if computed_integrity != schema_payload["integrity_summary"]:
        errors.append("integrity_summary.json does not match recomputed integrity")

    provenance = cast(dict[str, object], schema_payload["provenance"])
    errors.extend(
        _verify_provenance_consistency(
            export_dir=metadata_dir,
            records=records,
            artifact_hashes=artifact_hashes,
            provenance=provenance,
        )
    )

    computed_bundle_hash = compute_bundle_sha256(
        artifact_hashes=artifact_hashes,
        manifest_rows=cast(list[dict[str, object]], manifest_rows),
        schema_payload=export_schema_catalog(),
    )
    artifact_index = cast(dict[str, Any], schema_payload["artifact_index"])
    if artifact_index.get("bundle_sha256") != computed_bundle_hash:
        errors.append("artifact_index bundle_sha256 mismatch")

    missing_states = [
        (record.relative_path, record.status)
        for record in records
        if record.status in {"missing_required", "missing_recommended", "copy_failed"}
    ]

    return {
        "valid": not errors and integrity.passed,
        "errors": errors,
        "bundle_sha256": computed_bundle_hash,
        "integrity_summary": computed_integrity,
        "reproducibility_stamp": schema_payload["reproducibility_stamp"],
        "run_metadata": schema_payload["run_metadata"],
        "artifact_index": schema_payload["artifact_index"],
        "missing_states": missing_states,
    }
