from __future__ import annotations

from collections.abc import Mapping

REQUIRED_KEYS = (
    "sample_name",
    "sample_type",
    "source_type",
    "generation_timestamp",
    "random_seed",
    "files_written",
    "row_counts",
    "intended_usage",
    "compatibility_notes",
    "expected_outputs_written",
    "freeze_command_used",
    "freeze_timestamp",
)


def validate_manifest(payload: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in payload:
            errors.append(f"missing required key: {key}")

    if "files_written" in payload and not isinstance(payload.get("files_written"), list):
        errors.append("files_written must be a list")
    if "row_counts" in payload and not isinstance(payload.get("row_counts"), dict):
        errors.append("row_counts must be an object")
    if "intended_usage" in payload and not isinstance(payload.get("intended_usage"), list):
        errors.append("intended_usage must be a list")
    if "compatibility_notes" in payload and not isinstance(payload.get("compatibility_notes"), list):
        errors.append("compatibility_notes must be a list")
    if "expected_outputs_written" in payload and not isinstance(payload.get("expected_outputs_written"), list):
        errors.append("expected_outputs_written must be a list")
    if "freeze_command_used" in payload and not isinstance(payload.get("freeze_command_used"), list):
        errors.append("freeze_command_used must be a list")
    return errors


def ensure_manifest(payload: Mapping[str, object]) -> None:
    errors = validate_manifest(payload)
    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"manifest validation failed: {joined}")

