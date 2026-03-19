from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from arena.artifacts.integrity import verify_artifact_bundle


def _dict_val(d: dict[str, object], key: str) -> dict[str, Any]:
    """Extract a dict value, defaulting to empty dict."""
    v = d.get(key, {})
    return cast(dict[str, Any], v) if isinstance(v, dict) else {}


def replay_artifact_bundle(bundle_path: Path) -> int:
    result = verify_artifact_bundle(bundle_path)
    print(f"artifact_bundle: {bundle_path.resolve()}")
    print(f"valid: {int(bool(result['valid']))}")

    repro = _dict_val(result, "reproducibility_stamp")
    if repro:
        print("reproducibility_metadata:")
        for key in [
            "timestamp",
            "python_version",
            "platform",
            "git_commit",
            "artifact_subsystem_version",
            "export_mode",
            "deterministic_flag",
            "policy_version",
        ]:
            print(f"- {key}: {repro.get(key, '')}")

    run_metadata = _dict_val(result, "run_metadata")
    if run_metadata:
        print("run_metadata:")
        for key in [
            "run_id",
            "timestamp",
            "hostname",
            "artifact_count",
            "missing_required_count",
            "missing_recommended_count",
            "excluded_count",
        ]:
            print(f"- {key}: {run_metadata.get(key, '')}")

    integrity = _dict_val(result, "integrity_summary")
    print("integrity_summary:")
    for key in [
        "copied_records",
        "duplicate_output_paths",
        "missing_artifacts",
        "hash_mismatches",
        "validated_hash_entries",
        "passed",
    ]:
        print(f"- {key}: {integrity.get(key, '')}")

    missing_states = result.get("missing_states", [])
    print("missing_artifacts:")
    if isinstance(missing_states, list) and missing_states:
        for relative_path, status in missing_states:
            print(f"- {relative_path} ({status})")
    else:
        print("- (none)")

    errors = result.get("errors", [])
    if isinstance(errors, list) and errors:
        print("verification_errors:")
        for error in errors:
            print(f"- {error}")

    return 0 if result["valid"] else 1
