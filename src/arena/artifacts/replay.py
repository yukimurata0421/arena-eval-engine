from __future__ import annotations

from pathlib import Path

from arena.artifacts.integrity import verify_artifact_bundle


def replay_artifact_bundle(bundle_path: Path) -> int:
    result = verify_artifact_bundle(bundle_path)
    print(f"artifact_bundle: {bundle_path.resolve()}")
    print(f"valid: {int(result['valid'])}")

    repro = result.get("reproducibility_stamp", {})
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

    run_metadata = result.get("run_metadata", {})
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

    integrity = result.get("integrity_summary", {})
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
    if missing_states:
        for relative_path, status in missing_states:
            print(f"- {relative_path} ({status})")
    else:
        print("- (none)")

    errors = result.get("errors", [])
    if errors:
        print("verification_errors:")
        for error in errors:
            print(f"- {error}")

    return 0 if result["valid"] else 1

