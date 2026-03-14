from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.tools.artifacts import app, discovery, documentation, manifest, packaging, selection
from scripts.tools.artifacts.models import AICandidateFile


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _build_artifact_fixture(base_dir: Path) -> dict[str, Path]:
    files = {
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
    return files


def _patch_artifact_subsystem(monkeypatch, base_dir: Path) -> list[AICandidateFile]:
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
    return custom_candidates


def test_ai_export_end_to_end_generates_artifacts_and_states(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "fixture_output"
    output_root = tmp_path / "exports"
    _build_artifact_fixture(base_dir)
    _patch_artifact_subsystem(monkeypatch, base_dir)

    export_dir, records = app.export_ai_folder(base_dir=base_dir, output_root=output_root)

    assert export_dir.exists()
    assert len(records) == 12

    manifest_path = export_dir / "ai_selected_manifest.csv"
    extended_manifest_path = export_dir / "ai_selected_manifest_extended.csv"
    candidate_status_path = export_dir / "ai_export_candidate_status.csv"
    summary_path = export_dir / "ai_export_summary.txt"
    hashes_path = export_dir / "artifact_hashes.txt"
    repro_stamp_path = export_dir / "reproducibility_stamp.json"
    provenance_path = export_dir / "artifact_provenance.json"
    run_metadata_path = export_dir / "run_metadata.json"
    lineage_path = export_dir / "artifact_lineage.json"
    integrity_summary_path = export_dir / "integrity_summary.json"
    artifact_index_path = export_dir / "artifact_index.json"
    settings_md_path = export_dir / "ai_settings_summary.md"
    settings_json_path = export_dir / "ai_settings_snapshot.json"
    change_point_path = export_dir / "ai_change_point_settings_note.md"
    hardware_path = export_dir / "ai_hardware_date_recommendation.md"
    methodology_path = export_dir / "analysis_methodology.md"
    analysis_design_path = export_dir / "analysis_design.md"
    needed_files_path = export_dir / "ai_needed_files_for_statistics.md"
    gemini_pack_manifest = export_dir / "for_gemini" / "pack_manifest.txt"
    gpt_pack_manifest = export_dir / "for_GPT" / "pack_manifest.txt"
    grok_pack_manifest = export_dir / "for_grok" / "pack_manifest.txt"

    for path in [
        manifest_path,
        extended_manifest_path,
        candidate_status_path,
        summary_path,
        hashes_path,
        repro_stamp_path,
        provenance_path,
        run_metadata_path,
        lineage_path,
        integrity_summary_path,
        artifact_index_path,
        settings_md_path,
        settings_json_path,
        change_point_path,
        hardware_path,
        methodology_path,
        analysis_design_path,
        needed_files_path,
        gemini_pack_manifest,
        gpt_pack_manifest,
        grok_pack_manifest,
    ]:
        assert path.exists(), path

    manifest_rows = _read_csv_rows(manifest_path)
    statuses = {row["relative_path"]: row["status"] for row in manifest_rows}
    assert statuses["reports/shared_report.txt"] == "ok"
    assert statuses["required/primary.csv"] == "ok"
    assert statuses["required/missing_required.txt"] == "missing_required"
    assert statuses["recommended/missing_recommended.txt"] == "missing_recommended"
    assert statuses["datasets/dist_1m_minute_raw_sample.txt"] == "excluded_by_rule"
    assert statuses["aliases/shared_alias.txt"] == "duplicate_source_skipped"
    assert statuses["copy_fail/copy_fail.txt"] == "copy_failed"
    assert statuses["performance/phase_config_daily_mapping.csv"] == "ok"
    assert statuses["discovered/power_analysis_summary.csv"] == "ok"

    fallback_row = next(row for row in manifest_rows if row["relative_path"] == "performance/phase_config_daily_mapping.csv")
    assert "matched_by_glob=" in fallback_row["note"]
    hashed_row = next(row for row in manifest_rows if row["relative_path"] == "reports/shared_report.txt")
    assert len(hashed_row["artifact_sha256"]) == 64

    candidate_rows = {row["logical_name"]: row for row in _read_csv_rows(candidate_status_path)}
    assert candidate_rows["candidate_selected"]["exists"] == "1"
    assert candidate_rows["candidate_selected"]["copied_to_export"] == "1"
    assert "manifest_status=excluded_by_rule" in candidate_rows["candidate_excluded"]["note"]
    assert "manifest_status=copy_failed" in candidate_rows["candidate_copy_failed"]["note"]
    assert candidate_rows["candidate_copy_failed"]["copied_to_export"] == "0"
    assert candidate_rows["candidate_missing"]["exists"] == "0"
    assert "missing candidate" in candidate_rows["candidate_missing"]["note"]
    assert "matched_by_glob=" in candidate_rows["phase_config_daily_mapping"]["note"]

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "integrity_passed: 1" in summary_text
    assert "artifact_hashes_txt:" in summary_text
    assert "reproducibility_stamp_json:" in summary_text
    assert "artifact_provenance_json:" in summary_text
    assert "run_metadata_json:" in summary_text
    assert "artifact_lineage_json:" in summary_text
    assert "integrity_summary_json:" in summary_text
    assert "artifact_index_json:" in summary_text
    assert "bundle_sha256:" in summary_text
    assert "required/missing_required.txt (missing_required)" in summary_text
    assert "recommended/missing_recommended.txt (missing_recommended)" in summary_text
    assert "datasets/dist_1m_minute_raw_sample.txt (excluded_by_rule)" in summary_text
    assert "aliases/shared_alias.txt" in summary_text

    assert "# Analysis Methodology" in methodology_path.read_text(encoding="utf-8")
    assert "## settings.toml snapshot" in settings_md_path.read_text(encoding="utf-8")
    assert "# Change Point Configuration Note" in change_point_path.read_text(encoding="utf-8")
    assert "# Hardware Date Boundary Recommendation" in hardware_path.read_text(encoding="utf-8")
    analysis_design_text = analysis_design_path.read_text(encoding="utf-8")
    assert "## Hardware Date Recommendation" in analysis_design_text
    assert "## Change Point Settings Note" in analysis_design_text

    assert "profile: for_gemini" in gemini_pack_manifest.read_text(encoding="utf-8")
    assert "profile: for_GPT" in gpt_pack_manifest.read_text(encoding="utf-8")
    assert "profile: for_grok" in grok_pack_manifest.read_text(encoding="utf-8")
    assert "files/shared_report.txt" in hashes_path.read_text(encoding="utf-8")
    assert "\"artifact_subsystem_version\"" in repro_stamp_path.read_text(encoding="utf-8")
    assert json.loads(provenance_path.read_text(encoding="utf-8"))["entries"]
    assert json.loads(run_metadata_path.read_text(encoding="utf-8"))["artifact_count"] >= 1
    assert json.loads(lineage_path.read_text(encoding="utf-8"))["edges"]
    assert json.loads(integrity_summary_path.read_text(encoding="utf-8"))["passed"] is True
    assert len(json.loads(artifact_index_path.read_text(encoding="utf-8"))["bundle_sha256"]) == 64


def test_selection_fallback_and_duplicate_results_are_stable(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "fixture_output"
    output_root = tmp_path / "exports"
    fixture_paths = _build_artifact_fixture(base_dir)
    _patch_artifact_subsystem(monkeypatch, base_dir)

    first_targets = selection.iter_ai_targets(base_dir)
    second_targets = selection.iter_ai_targets(base_dir)
    assert first_targets == second_targets
    assert first_targets == [
        ("reports/shared_report.txt", "required"),
        ("required/primary.csv", "required"),
        ("required/missing_required.txt", "required"),
        ("performance/phase_config_daily_mapping.csv", "required"),
        ("recommended/secondary.csv", "recommended"),
        ("recommended/missing_recommended.txt", "recommended"),
        ("datasets/dist_1m_minute_raw_sample.txt", "recommended"),
        ("aliases/shared_alias.txt", "recommended"),
        ("copy_fail/copy_fail.txt", "recommended"),
        ("candidates/candidate_selected.csv", "recommended"),
        ("candidates/missing_candidate.csv", "recommended"),
        ("discovered/power_analysis_summary.csv", "recommended"),
    ]

    resolved_first, note_first = discovery.resolve_ai_source_path(base_dir, "performance/phase_config_daily_mapping.csv")
    resolved_second, note_second = discovery.resolve_ai_source_path(base_dir, "performance/phase_config_daily_mapping.csv")
    assert resolved_first == resolved_second == fixture_paths["fallback_primary"]
    assert note_first == note_second

    export_dir_a, records_a = app.export_ai_folder(base_dir=base_dir, output_root=output_root / "run_a")
    export_dir_b, records_b = app.export_ai_folder(base_dir=base_dir, output_root=output_root / "run_b")
    assert export_dir_a != export_dir_b
    assert [
        (record.relative_path, record.status, record.source_path, record.note)
        for record in records_a
    ] == [
        (record.relative_path, record.status, record.source_path, record.note)
        for record in records_b
    ]


def test_legacy_wrapper_supports_dry_run(tmp_path: Path) -> None:
    base_dir = tmp_path / "wrapper_fixture"
    out_dir = tmp_path / "wrapper_out"
    _write_text(base_dir / "notes" / "result.txt", "wrapper compatible\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/merge_output_for_ai/merge_output_for_ai.py",
            "--base",
            str(base_dir),
            "--out",
            str(out_dir),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[INFO] dry-run enabled: skipping merged output generation" in result.stdout
    assert (out_dir / "manifest.csv").exists()
    assert not (out_dir / "merged_for_ai.md").exists()
