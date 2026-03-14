from __future__ import annotations

from copy import deepcopy

from jsonschema import validate

STATE_NAMES = [
    "ok",
    "missing_required",
    "missing_recommended",
    "excluded_by_rule",
    "duplicate_source_skipped",
    "copy_failed",
    "candidate_only_found",
    "missing_candidate",
]

MANIFEST_RECORD_SCHEMA = {
    "type": "object",
    "required": [
        "relative_path",
        "category",
        "required_level",
        "exists",
        "copied",
        "size_bytes",
        "mtime",
        "status",
        "note",
        "artifact_sha256",
    ],
    "properties": {
        "relative_path": {"type": "string", "minLength": 1},
        "category": {"type": "string"},
        "required_level": {"type": "string", "enum": ["required", "recommended", "candidate"]},
        "exists": {"type": "integer", "enum": [0, 1]},
        "copied": {"type": "integer", "enum": [0, 1]},
        "size_bytes": {"type": "integer", "minimum": 0},
        "mtime": {"type": "string"},
        "status": {"type": "string", "enum": STATE_NAMES},
        "note": {"type": "string"},
        "artifact_sha256": {"type": "string", "pattern": r"(^$|^[a-f0-9]{64}$)"},
    },
    "additionalProperties": True,
}

CANDIDATE_STATUS_SCHEMA = {
    "type": "object",
    "required": [
        "logical_name",
        "expected_path",
        "exists",
        "copied_to_export",
        "priority",
        "note",
    ],
    "properties": {
        "logical_name": {"type": "string", "minLength": 1},
        "expected_path": {"type": "string", "minLength": 1},
        "exists": {"type": "integer", "enum": [0, 1]},
        "copied_to_export": {"type": "integer", "enum": [0, 1]},
        "priority": {"type": "string", "enum": ["A", "B"]},
        "note": {"type": "string"},
    },
    "additionalProperties": True,
}

PACK_MANIFEST_SCHEMA = {
    "type": "object",
    "required": [
        "profile",
        "generated_at",
        "core_files",
        "detail_files",
        "core_file_names",
        "detail_file_examples",
    ],
    "properties": {
        "profile": {"type": "string", "minLength": 1},
        "generated_at": {"type": "string", "minLength": 1},
        "core_files": {"type": "integer", "minimum": 0},
        "detail_files": {"type": "integer", "minimum": 0},
        "core_file_names": {"type": "array", "items": {"type": "string"}},
        "detail_file_examples": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

INTEGRITY_SUMMARY_SCHEMA = {
    "type": "object",
    "required": [
        "duplicate_source_skipped",
        "duplicate_output_paths",
        "copied_records",
        "missing_artifacts",
        "hash_mismatches",
        "validated_hash_entries",
        "passed",
    ],
    "properties": {
        "duplicate_source_skipped": {"type": "integer", "minimum": 0},
        "duplicate_output_paths": {"type": "integer", "minimum": 0},
        "copied_records": {"type": "integer", "minimum": 0},
        "missing_artifacts": {"type": "integer", "minimum": 0},
        "hash_mismatches": {"type": "integer", "minimum": 0},
        "validated_hash_entries": {"type": "integer", "minimum": 0},
        "passed": {"type": "boolean"},
    },
    "additionalProperties": False,
}

ARTIFACT_PROVENANCE_ENTRY_SCHEMA = {
    "type": "object",
    "required": [
        "artifact_path",
        "artifact_sha256",
        "source_paths",
        "generation_stage",
        "generation_timestamp",
        "policy_version",
    ],
    "properties": {
        "artifact_path": {"type": "string", "minLength": 1},
        "artifact_sha256": {"type": "string", "pattern": r"^[a-f0-9]{64}$"},
        "source_paths": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "generation_stage": {"type": "string", "minLength": 1},
        "generation_timestamp": {"type": "string", "minLength": 1},
        "policy_version": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}

ARTIFACT_PROVENANCE_SCHEMA = {
    "type": "object",
    "required": ["generated_at", "policy_version", "entries"],
    "properties": {
        "generated_at": {"type": "string", "minLength": 1},
        "policy_version": {"type": "string", "minLength": 1},
        "entries": {
            "type": "array",
            "items": ARTIFACT_PROVENANCE_ENTRY_SCHEMA,
        },
    },
    "additionalProperties": False,
}

RUN_METADATA_SCHEMA = {
    "type": "object",
    "required": [
        "run_id",
        "timestamp",
        "hostname",
        "python_version",
        "platform",
        "git_commit",
        "deterministic_flag",
        "artifact_count",
        "missing_required_count",
        "missing_recommended_count",
        "excluded_count",
    ],
    "properties": {
        "run_id": {"type": "string", "minLength": 1},
        "timestamp": {"type": "string", "minLength": 1},
        "hostname": {"type": "string"},
        "python_version": {"type": "string", "minLength": 1},
        "platform": {"type": "string", "minLength": 1},
        "git_commit": {"type": "string"},
        "deterministic_flag": {"type": "boolean"},
        "artifact_count": {"type": "integer", "minimum": 0},
        "missing_required_count": {"type": "integer", "minimum": 0},
        "missing_recommended_count": {"type": "integer", "minimum": 0},
        "excluded_count": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}

LINEAGE_NODE_SCHEMA = {
    "type": "object",
    "required": ["id", "path", "node_kind", "category", "status"],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "path": {"type": "string", "minLength": 1},
        "node_kind": {"type": "string", "enum": ["artifact", "source"]},
        "category": {"type": "string", "minLength": 1},
        "status": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}

LINEAGE_EDGE_SCHEMA = {
    "type": "object",
    "required": ["from", "to", "relationship"],
    "properties": {
        "from": {"type": "string", "minLength": 1},
        "to": {"type": "string", "minLength": 1},
        "relationship": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}

ARTIFACT_LINEAGE_SCHEMA = {
    "type": "object",
    "required": ["generated_at", "policy_version", "nodes", "edges", "artifact_types"],
    "properties": {
        "generated_at": {"type": "string", "minLength": 1},
        "policy_version": {"type": "string", "minLength": 1},
        "nodes": {"type": "array", "items": LINEAGE_NODE_SCHEMA},
        "edges": {"type": "array", "items": LINEAGE_EDGE_SCHEMA},
        "artifact_types": {
            "type": "object",
            "additionalProperties": {"type": "string", "minLength": 1},
        },
    },
    "additionalProperties": False,
}

ARTIFACT_INDEX_SCHEMA = {
    "type": "object",
    "required": [
        "bundle_id",
        "timestamp",
        "artifact_count",
        "bundle_sha256",
        "run_metadata_ref",
        "provenance_ref",
        "integrity_ref",
    ],
    "properties": {
        "bundle_id": {"type": "string", "minLength": 1},
        "timestamp": {"type": "string", "minLength": 1},
        "artifact_count": {"type": "integer", "minimum": 0},
        "bundle_sha256": {"type": "string", "pattern": r"^[a-f0-9]{64}$"},
        "run_metadata_ref": {"type": "string", "minLength": 1},
        "provenance_ref": {"type": "string", "minLength": 1},
        "integrity_ref": {"type": "string", "minLength": 1},
        "lineage_ref": {"type": "string", "minLength": 1},
        "manifest_ref": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}


def validate_manifest_record_rows(rows: list[dict[str, object]]) -> None:
    for row in rows:
        validate(instance=row, schema=MANIFEST_RECORD_SCHEMA)


def validate_candidate_status_rows(rows: list[dict[str, object]]) -> None:
    for row in rows:
        validate(instance=row, schema=CANDIDATE_STATUS_SCHEMA)


def validate_pack_manifest_payload(payload: dict[str, object]) -> None:
    validate(instance=payload, schema=PACK_MANIFEST_SCHEMA)


def validate_integrity_summary(payload: dict[str, object]) -> None:
    validate(instance=payload, schema=INTEGRITY_SUMMARY_SCHEMA)


def validate_artifact_provenance(payload: dict[str, object]) -> None:
    validate(instance=payload, schema=ARTIFACT_PROVENANCE_SCHEMA)


def validate_run_metadata(payload: dict[str, object]) -> None:
    validate(instance=payload, schema=RUN_METADATA_SCHEMA)


def validate_artifact_lineage(payload: dict[str, object]) -> None:
    validate(instance=payload, schema=ARTIFACT_LINEAGE_SCHEMA)


def validate_artifact_index(payload: dict[str, object]) -> None:
    validate(instance=payload, schema=ARTIFACT_INDEX_SCHEMA)


def export_schema_catalog() -> dict[str, object]:
    return {
        "state_names": list(STATE_NAMES),
        "manifest_record": deepcopy(MANIFEST_RECORD_SCHEMA),
        "candidate_status": deepcopy(CANDIDATE_STATUS_SCHEMA),
        "pack_manifest": deepcopy(PACK_MANIFEST_SCHEMA),
        "integrity_summary": deepcopy(INTEGRITY_SUMMARY_SCHEMA),
        "artifact_provenance": deepcopy(ARTIFACT_PROVENANCE_SCHEMA),
        "run_metadata": deepcopy(RUN_METADATA_SCHEMA),
        "artifact_lineage": deepcopy(ARTIFACT_LINEAGE_SCHEMA),
        "artifact_index": deepcopy(ARTIFACT_INDEX_SCHEMA),
    }

