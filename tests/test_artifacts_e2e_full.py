from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.tools.artifacts import app, discovery, documentation, manifest, packaging, selection
from scripts.tools.artifacts.integrity import run_ai_export_integrity_check
from scripts.tools.artifacts.models import AICandidateFile


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def _build_artifact_fixture(base_dir: Path) -> dict[str, Path]:
    return {
        "shared_report": _write_text(base_dir / "reports" / "shared_report.txt", "shared report\n"),
        "primary": _write_text(base_dir / "required" / "primary.csv", "day,value\n2026-01-14,1\n"),
        "secondary": _write_text(base_dir / "recommended" / "secondary.csv", "day,value\n2026-01-15,2\n"),
        "fallback_primary": _write_text(base_dir / "fallback" / "phase_config_daily_mapping.csv", "day,phase\n2026-01-14,A\n"),
        "fallback_shadow": _write_text(base_dir / "_shadow" / "phase_config_daily_mapping.csv", "day,phase\n2026-01-13,Z\n"),
        "selected_candidate": _write_text(base_dir / "candidates" / "candidate_selected.csv", "metric,value\nx,1\n"),
        "excluded_dataset": _write_text(base_dir / "datasets" / "dist_1m_minute_raw_sample.txt", "raw minute data\n"),
        "copy_fail": _write_text(base_dir / "copy_fail" / "copy_fail.txt", "copy failure target\n"),
        "discovered": _write_text(base_dir / "discovered" / "power_analysis_summary.csv", "power,0.8\n"),
        "settings": _write_text(base_dir / "config" / "settings.toml", "[quality]\nmin_auc_n_used = 7\n"),
    }


def _patch_artifact_subsystem(monkeypatch, base_dir: Path) -> None:
    custom_candidates = [
        AICandidateFile(
            logical_name="phase_config_daily_mapping",
            relative_path="performance/phase_config_daily_mapping.csv",
            expected_path="output/performance/phase_config_daily_mapping.csv",
            priority="A",
            reason="Fallback resolution should be stable.",
            what_it_enables="Transition-day review.",
            rationale="phase_label_verification",
        ),
        AICandidateFile(
            logical_name="candidate_selected",
            relative_path="candidates/candidate_selected.csv",
            expected_path="output/candidates/candidate_selected.csv",
            priority="B",
            reason="Confirms candidate artifacts are copied.",
            what_it_enables="Supplemental AI review.",
            rationale="candidate_selection",
        ),
        AICandidateFile(
            logical_name="candidate_excluded",
            relative_path="datasets/dist_1m_minute_raw_sample.txt",
            expected_path="output/datasets/dist_1m_minute_raw_sample.txt",
            priority="B",
            reason="Confirms exclusion is visible.",
            what_it_enables="Failure visibility for raw datasets.",
            rationale="exclude_rules",
        ),
        AICandidateFile(
            logical_name="candidate_copy_failed",
            relative_path="copy_fail/copy_fail.txt",
            expected_path="output/copy_fail/copy_fail.txt",
            priority="B",
            reason="Confirms copy failure remains visible.",
            what_it_enables="Filesystem failure auditing.",
            rationale="copy_failure_audit",
        ),
        AICandidateFile(
            logical_name="candidate_missing",
            relative_path="candidates/missing_candidate.csv",
            expected_path="output/candidates/missing_candidate.csv",
            priority="A",
            reason="Confirms missing candidate reporting.",
            what_it_enables="Future artifact planning.",
            rationale="missing_candidate_audit",
        ),
    ]

    monkeypatch.setattr(
        selection,
        "AI_REQUIRED_FILES",
        [
            "reports/shared_report.txt",
            "required/primary.csv",
            "required/missing_required.txt",
            "performance/phase_config_daily_mapping.csv",
        ],
    )
    monkeypatch.setattr(
        selection,
        "AI_RECOMMENDED_FILES",
        [
            "recommended/secondary.csv",
            "recommended/missing_recommended.txt",
            "datasets/dist_1m_minute_raw_sample.txt",
            "aliases/shared_alias.txt",
            "copy_fail/copy_fail.txt",
        ],
    )
    monkeypatch.setattr(selection, "get_ai_candidate_files", lambda: custom_candidates)
    monkeypatch.setattr(manifest, "get_ai_candidate_files", lambda: custom_candidates)
    monkeypatch.setattr(app, "get_ai_candidate_files", lambda: custom_candidates)

    monkeypatch.setattr(discovery, "_iter_ai_search_roots", lambda current_base_dir: [current_base_dir])
    monkeypatch.setattr(discovery, "_iter_ai_discovery_roots", lambda current_base_dir: [current_base_dir])
    monkeypatch.setattr(
        discovery,
        "AI_FILENAME_GLOB_FALLBACKS",
        {
            "performance/phase_config_daily_mapping.csv": ["phase_config_daily_mapping.csv"],
            "aliases/shared_alias.txt": ["shared_report.txt"],
        },
    )
    monkeypatch.setattr(discovery, "AI_FALLBACK_SOURCE_PATHS", {})
    monkeypatch.setattr(discovery, "AI_PRIORITY_B_DISCOVERY_INCLUDE_GLOBS", ["*power*analysis*.csv"])
    monkeypatch.setattr(discovery, "AI_PRIORITY_B_DISCOVERY_EXCLUDE_GLOBS", [])
    monkeypatch.setattr(discovery, "AI_PRIORITY_B_ALLOWED_EXTS", {".csv", ".txt"})
    monkeypatch.setattr(discovery, "AI_PRIORITY_B_MAX_FILE_BYTES", 1_000_000)

    monkeypatch.setattr(
        packaging,
        "AI_GEMINI_PACK_KEYS",
        ["ai_export_summary.txt", "ai_hardware_date_recommendation.md", "primary.csv"],
    )
    monkeypatch.setattr(packaging, "AI_GEMINI_CORE_LIMIT", 3)
    monkeypatch.setattr(
        packaging,
        "AI_GPT_PACK_KEYS",
        ["primary.csv", "phase_config_daily_mapping.csv", "analysis_methodology.md", "shared_report.txt"],
    )
    monkeypatch.setattr(packaging, "AI_GPT_CORE_LIMIT", 4)

    settings_stub = SimpleNamespace(
        path=str(base_dir / "config" / "settings.toml"),
        data={"quality": {"min_auc_n_used": 7, "min_minutes_covered": 30}},
    )
    monkeypatch.setattr(documentation, "load_settings", lambda force_reload=True: settings_stub)
    monkeypatch.setattr(documentation, "get_quality_thresholds", lambda: (7, 30))
    monkeypatch.setattr(app, "load_settings", lambda force_reload=False: settings_stub)

    real_copy2 = shutil.copy2

    def controlled_copy2(src: str | Path, dst: str | Path, *args, **kwargs):
        if Path(src).name == "copy_fail.txt":
            raise PermissionError("synthetic copy failure for tests")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(manifest.shutil, "copy2", controlled_copy2)


def test_deterministic_mode_produces_identical_core_outputs(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "fixture_output"
    output_root = tmp_path / "exports"
    _build_artifact_fixture(base_dir)
    _patch_artifact_subsystem(monkeypatch, base_dir)

    export_dir_a, _ = app.export_ai_folder(base_dir=base_dir, output_root=output_root / "run_a", deterministic=True)
    export_dir_b, _ = app.export_ai_folder(base_dir=base_dir, output_root=output_root / "run_b", deterministic=True)

    comparable_files = [
        "artifact_hashes.txt",
        "reproducibility_stamp.json",
        "artifact_provenance.json",
        "run_metadata.json",
        "artifact_lineage.json",
        "integrity_summary.json",
        "artifact_index.json",
        "ai_export_summary.txt",
        "ai_selected_manifest.csv",
        "ai_selected_manifest_extended.csv",
        "for_GPT/pack_manifest.txt",
        "for_gemini/pack_manifest.txt",
        "for_grok/pack_manifest.txt",
    ]
    for relative_path in comparable_files:
        assert (export_dir_a / relative_path).read_text(encoding="utf-8") == (
            export_dir_b / relative_path
        ).read_text(encoding="utf-8")

    repro_payload = json.loads((export_dir_a / "reproducibility_stamp.json").read_text(encoding="utf-8"))
    assert repro_payload["timestamp"] == "1970-01-01T00:00:00"
    assert repro_payload["deterministic_flag"] is True
    assert repro_payload["export_mode"] == "ai_export"
    artifact_index = json.loads((export_dir_a / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["bundle_sha256"] == artifact_index["bundle_id"]


def test_integrity_detects_hash_mismatch_and_missing_artifact(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "fixture_output"
    output_root = tmp_path / "exports"
    _build_artifact_fixture(base_dir)
    _patch_artifact_subsystem(monkeypatch, base_dir)

    export_dir, records = app.export_ai_folder(base_dir=base_dir, output_root=output_root, deterministic=True)
    (export_dir / "files" / "shared_report.txt").write_text("tampered\n", encoding="utf-8", newline="\n")
    (export_dir / "files" / "primary.csv").unlink()

    integrity = run_ai_export_integrity_check(
        records,
        export_dir=export_dir,
        hash_manifest_path=export_dir / "artifact_hashes.txt",
    )

    assert integrity.passed is False
    assert integrity.hash_mismatches >= 1
    assert integrity.missing_artifacts >= 1
    assert integrity.validated_hash_entries >= 1


def test_verify_and_replay_cli_validate_bundle(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "fixture_output"
    output_root = tmp_path / "exports"
    _build_artifact_fixture(base_dir)
    _patch_artifact_subsystem(monkeypatch, base_dir)

    export_dir, _ = app.export_ai_folder(base_dir=base_dir, output_root=output_root, deterministic=True)
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}

    verify_result = subprocess.run(
        [sys.executable, "-m", "arena.cli", "artifacts", "verify", str(export_dir)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert verify_result.returncode == 0, verify_result.stderr
    assert "valid: 1" in verify_result.stdout
    assert "bundle_sha256:" in verify_result.stdout

    replay_result = subprocess.run(
        [sys.executable, "-m", "arena.cli", "artifacts", "replay", str(export_dir)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert replay_result.returncode == 0, replay_result.stderr
    assert "reproducibility_metadata:" in replay_result.stdout
    assert "run_metadata:" in replay_result.stdout
    assert "missing_artifacts:" in replay_result.stdout
    assert "- required/missing_required.txt (missing_required)" in replay_result.stdout


def test_verify_cli_fails_on_provenance_or_hash_drift(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "fixture_output"
    output_root = tmp_path / "exports"
    _build_artifact_fixture(base_dir)
    _patch_artifact_subsystem(monkeypatch, base_dir)

    export_dir, _ = app.export_ai_folder(base_dir=base_dir, output_root=output_root, deterministic=True)
    provenance_path = export_dir / "artifact_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["entries"][0]["source_paths"] = ["mismatched-source"]
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}

    result = subprocess.run(
        [sys.executable, "-m", "arena.cli", "artifacts", "verify", str(export_dir)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "artifact_provenance" in result.stdout
