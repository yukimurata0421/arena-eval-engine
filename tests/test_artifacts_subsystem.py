from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.tools.artifacts.cli import build_parser
from scripts.tools.artifacts.integrity import run_ai_export_integrity_check
from scripts.tools.artifacts.schema import (
    validate_artifact_index,
    validate_artifact_lineage,
    validate_artifact_provenance,
    validate_integrity_summary,
    validate_manifest_record_rows,
    validate_pack_manifest_payload,
    validate_run_metadata,
)
from scripts.tools.artifacts.manifest import write_ai_selected_manifest_extended
from scripts.tools.artifacts.models import AICandidateStatus, AIManifestRecord

from jsonschema import ValidationError


def test_artifacts_cli_keeps_legacy_options() -> None:
    parser = build_parser()
    args = parser.parse_args(["--export-ai-folder", "--ai-export-root", "X:\\tmp", "--deterministic"])
    assert args.export_ai_folder is True
    assert args.ai_export_root == "X:\\tmp"
    assert args.ai_export_use_out_parent is False
    assert args.deterministic is True


def test_run_ai_export_integrity_check_detects_duplicate_output_paths() -> None:
    records = [
        AIManifestRecord(
            relative_path="a.txt",
            category="misc",
            required_level="required",
            exists=True,
            copied=True,
            size_bytes=1,
            mtime="2026-01-01T00:00:00",
            status="ok",
            note="",
            copied_path="files/shared.txt",
        ),
        AIManifestRecord(
            relative_path="b.txt",
            category="misc",
            required_level="recommended",
            exists=True,
            copied=True,
            size_bytes=1,
            mtime="2026-01-01T00:00:00",
            status="ok",
            note="",
            copied_path="files/shared.txt",
        ),
    ]

    integrity = run_ai_export_integrity_check(records)

    assert integrity.passed is False
    assert integrity.duplicate_output_paths == 1
    assert integrity.copied_records == 2
    assert integrity.missing_artifacts == 0
    assert integrity.hash_mismatches == 0


def test_write_ai_selected_manifest_extended_includes_candidate_only_rows(tmp_path: Path) -> None:
    manifest_path = tmp_path / "ai_selected_manifest_extended.csv"
    records = [
        AIManifestRecord(
            relative_path="adsb_daily_summary_v2.csv",
            category="adsb",
            required_level="required",
            exists=True,
            copied=True,
            size_bytes=10,
            mtime="2026-01-01T00:00:00",
            status="ok",
            note="",
        )
    ]
    candidate_statuses = [
        AICandidateStatus(
            logical_name="pipeline_runs",
            relative_path="performance/pipeline_runs.jsonl",
            expected_path="output/performance/pipeline_runs.jsonl",
            exists=False,
            copied_to_export=False,
            priority="A",
            note="missing candidate",
            source_path="",
            reason="reason",
            what_it_enables="enables",
            rationale="hardware_transition_timing",
        )
    ]

    write_ai_selected_manifest_extended(records, candidate_statuses, manifest_path)

    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    candidate_only = [row for row in rows if row["relative_path"] == "performance/pipeline_runs.jsonl"]
    assert len(candidate_only) == 1
    assert candidate_only[0]["candidate_only"] == "1"
    assert candidate_only[0]["status"] == "missing_candidate"


def test_manifest_schema_validation_rejects_invalid_hash() -> None:
    with pytest.raises(ValidationError):
        validate_manifest_record_rows(
            [
                {
                    "relative_path": "x.csv",
                    "category": "misc",
                    "required_level": "required",
                    "exists": 1,
                    "copied": 1,
                    "size_bytes": 1,
                    "mtime": "2026-01-01T00:00:00",
                    "status": "ok",
                    "note": "",
                    "artifact_sha256": "not-a-sha256",
                }
            ]
        )


def test_pack_manifest_schema_validation_requires_core_names() -> None:
    with pytest.raises(ValidationError):
        validate_pack_manifest_payload(
            {
                "profile": "for_GPT",
                "generated_at": "2026-01-01T00:00:00",
                "core_files": 1,
                "detail_files": 0,
                "detail_file_examples": [],
            }
        )


def test_integrity_schema_validation_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        validate_integrity_summary(
            {
                "duplicate_source_skipped": 0,
                "duplicate_output_paths": -1,
                "copied_records": 1,
                "missing_artifacts": 0,
                "hash_mismatches": 0,
                "validated_hash_entries": 0,
                "passed": False,
            }
        )


def test_provenance_schema_validation_requires_hash() -> None:
    with pytest.raises(ValidationError):
        validate_artifact_provenance(
            {
                "generated_at": "2026-01-01T00:00:00",
                "policy_version": "1.0",
                "entries": [
                    {
                        "artifact_path": "files/x.csv",
                        "artifact_sha256": "",
                        "source_paths": ["C:/tmp/x.csv"],
                        "generation_stage": "ai_export_copy",
                        "generation_timestamp": "2026-01-01T00:00:00",
                        "policy_version": "1.0",
                    }
                ],
            }
        )


def test_run_metadata_schema_validation_requires_counts() -> None:
    with pytest.raises(ValidationError):
        validate_run_metadata(
            {
                "run_id": "bundle",
                "timestamp": "2026-01-01T00:00:00",
                "hostname": "host",
                "python_version": "3.11.0",
                "platform": "Windows",
                "git_commit": "",
                "deterministic_flag": True,
                "artifact_count": 1,
                "missing_required_count": 0,
                "excluded_count": 0,
            }
        )


def test_lineage_schema_validation_requires_edges_shape() -> None:
    with pytest.raises(ValidationError):
        validate_artifact_lineage(
            {
                "generated_at": "2026-01-01T00:00:00",
                "policy_version": "1.0",
                "nodes": [],
                "edges": [{"from": "a", "to": "b"}],
                "artifact_types": {},
            }
        )


def test_artifact_index_schema_validation_requires_sha256() -> None:
    with pytest.raises(ValidationError):
        validate_artifact_index(
            {
                "bundle_id": "bundle",
                "timestamp": "2026-01-01T00:00:00",
                "artifact_count": 1,
                "bundle_sha256": "short",
                "run_metadata_ref": "run_metadata.json",
                "provenance_ref": "artifact_provenance.json",
                "integrity_ref": "integrity_summary.json",
            }
        )
