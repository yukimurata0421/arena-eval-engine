from __future__ import annotations

import json
from pathlib import Path

from test_artifacts_e2e_full import _build_artifact_fixture, _patch_artifact_subsystem

from arena.artifacts.integrity import verify_artifact_bundle
from arena.artifacts.replay import replay_artifact_bundle
from scripts.tools.artifacts import app


def _export_bundle(tmp_path: Path, monkeypatch) -> Path:
    base_dir = tmp_path / "fixture_output"
    output_root = tmp_path / "exports"
    _build_artifact_fixture(base_dir)
    _patch_artifact_subsystem(monkeypatch, base_dir)
    export_dir, _records = app.export_ai_folder(base_dir=base_dir, output_root=output_root, deterministic=True)
    return export_dir


def test_integrity_detects_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    tampered = export_dir / "files" / "shared_report.txt"
    tampered.write_text("tampered\n", encoding="utf-8", newline="\n")

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert result["integrity_summary"]["hash_mismatches"] >= 1
    assert "integrity_summary.json does not match recomputed integrity" in result["errors"]


def test_integrity_reports_schema_validation_failure(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    run_metadata_path = export_dir / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    run_metadata.pop("missing_required_count")
    run_metadata_path.write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert any("run_metadata.json invalid:" in error for error in result["errors"])
    assert result["integrity_summary"]["passed"] is True


def test_integrity_reports_provenance_inconsistency(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    provenance_path = export_dir / "artifact_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["entries"][0]["source_paths"] = ["mismatched-source"]
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert any("artifact_provenance missing manifest source path" in error for error in result["errors"])
    assert result["run_metadata"]["artifact_count"] >= 1


def test_integrity_handles_missing_bundle_members_gracefully(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    (export_dir / "run_metadata.json").unlink()
    (export_dir / "for_GPT" / "pack_manifest.txt").unlink()

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert "run_metadata.json is missing" in result["errors"]
    assert "for_GPT/pack_manifest.txt is missing" in result["errors"]
    assert result["integrity_summary"]["passed"] is True
    assert result["missing_states"]


def test_integrity_handles_malformed_hash_manifest_input(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    (export_dir / "artifact_hashes.txt").write_text("not-a-valid-hash-line\n", encoding="utf-8", newline="\n")

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert any("artifact_hashes.txt invalid:" in error for error in result["errors"])
    assert result["integrity_summary"]["validated_hash_entries"] == 0


def test_integrity_reports_non_object_run_metadata_json(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    (export_dir / "run_metadata.json").write_text("[1, 2, 3]\n", encoding="utf-8", newline="\n")

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert any("run_metadata.json invalid: expected JSON object" in error for error in result["errors"])
    assert result["run_metadata"] == {}


def test_integrity_reports_non_object_repro_stamp_json(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    (export_dir / "reproducibility_stamp.json").write_text("[\"bad\"]\n", encoding="utf-8", newline="\n")

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert any("reproducibility_stamp.json invalid: expected JSON object" in error for error in result["errors"])
    assert result["reproducibility_stamp"] == {}


def test_integrity_reports_provenance_non_object_entry(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    provenance_path = export_dir / "artifact_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["entries"] = [provenance["entries"][0], "bad-entry"]
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr("arena.artifacts.integrity.validate_artifact_provenance", lambda payload: None)

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert any("artifact_provenance contains non-object entry" in error for error in result["errors"])


def test_integrity_handles_pack_manifest_malformed_line(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    pack_manifest = export_dir / "for_gemini" / "pack_manifest.txt"
    pack_manifest.write_text("profile: for_gemini\nbroken-line-without-colon\n", encoding="utf-8", newline="\n")

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert any("for_gemini/pack_manifest.txt invalid:" in error for error in result["errors"])
    assert result["integrity_summary"]["passed"] is True


def test_integrity_handles_hash_manifest_extra_malformed_lines(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    valid_line = (export_dir / "artifact_hashes.txt").read_text(encoding="utf-8").splitlines()[0]
    (export_dir / "artifact_hashes.txt").write_text(
        valid_line + "\nmalformed trailing line\n",
        encoding="utf-8",
        newline="\n",
    )

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert any("artifact_hashes.txt invalid:" in error for error in result["errors"])
    assert result["integrity_summary"]["validated_hash_entries"] == 0


def test_integrity_reports_hash_disagreement(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    provenance_path = export_dir / "artifact_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["entries"][0]["artifact_sha256"] = "0" * 64
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr("arena.artifacts.integrity.validate_artifact_provenance", lambda payload: None)

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert any("artifact_provenance hash mismatch:" in error for error in result["errors"])
    assert any("artifact_provenance disagrees with manifest hash:" in error for error in result["errors"])
    assert result["integrity_summary"]["passed"] is True


def test_integrity_reports_missing_copied_artifact(tmp_path: Path, monkeypatch) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    (export_dir / "files" / "shared_report.txt").unlink()

    result = verify_artifact_bundle(export_dir)

    assert result["valid"] is False
    assert result["integrity_summary"]["missing_artifacts"] >= 1
    assert any("artifact_provenance artifact missing on disk: files/shared_report.txt" in error for error in result["errors"])
    assert "integrity_summary.json does not match recomputed integrity" in result["errors"]


def test_replay_reports_missing_bundle(tmp_path: Path, capsys) -> None:
    missing_bundle = tmp_path / "missing_bundle"

    rc = replay_artifact_bundle(missing_bundle)

    captured = capsys.readouterr()
    assert rc == 1
    assert "valid: 0" in captured.out
    assert "verification_errors:" in captured.out
    assert "artifact bundle not found:" in captured.out


def test_replay_reports_missing_required_metadata(tmp_path: Path, monkeypatch, capsys) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    (export_dir / "run_metadata.json").unlink()

    rc = replay_artifact_bundle(export_dir)

    captured = capsys.readouterr()
    assert rc == 1
    assert "verification_errors:" in captured.out
    assert "run_metadata.json is missing" in captured.out
    assert "missing_artifacts:" in captured.out


def test_replay_reports_missing_artifacts(tmp_path: Path, monkeypatch, capsys) -> None:
    export_dir = _export_bundle(tmp_path, monkeypatch)
    (export_dir / "files" / "primary.csv").unlink()

    rc = replay_artifact_bundle(export_dir)

    captured = capsys.readouterr()
    assert rc == 1
    assert "integrity_summary:" in captured.out
    assert "- missing_artifacts:" in captured.out
    assert "- required/missing_required.txt (missing_required)" in captured.out
    assert "verification_errors:" in captured.out


def test_replay_handles_partial_reproducibility_metadata(monkeypatch, capsys, tmp_path: Path) -> None:
    expected = {
        "valid": False,
        "errors": ["synthetic failure"],
        "bundle_sha256": "",
        "integrity_summary": {
            "copied_records": 2,
            "duplicate_output_paths": 0,
            "missing_artifacts": 0,
            "hash_mismatches": 0,
            "validated_hash_entries": 0,
            "passed": False,
        },
        "reproducibility_stamp": {
            "timestamp": "2026-01-01T00:00:00",
            "git_commit": "abc123",
        },
        "run_metadata": {},
        "artifact_index": {},
        "missing_states": [],
    }
    monkeypatch.setattr("arena.artifacts.replay.verify_artifact_bundle", lambda bundle_path: expected)

    rc = replay_artifact_bundle(tmp_path / "bundle")

    captured = capsys.readouterr()
    assert rc == 1
    assert "- timestamp: 2026-01-01T00:00:00" in captured.out
    assert "- git_commit: abc123" in captured.out
    assert "- artifact_subsystem_version: " in captured.out
