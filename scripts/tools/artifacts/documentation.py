from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from arena.lib.config import get_quality_thresholds
from arena.lib.runtime_config import load_settings

from arena.artifacts.models import AICandidateFile, AICandidateStatus, AIExportIntegrity, AIManifestRecord
from arena.artifacts.policies import (
    AI_ANALYSIS_DESIGN_MD_FILENAME,
    AI_ANALYSIS_METHODOLOGY_MD_FILENAME,
    AI_ARTIFACT_HASHES_FILENAME,
    AI_CHANGE_POINT_NOTE_MD_FILENAME,
    AI_EXPORT_EXCLUDE_GLOB_PATTERNS,
    AI_HARDWARE_DATE_RECOMMENDATION_MD_FILENAME,
    AI_NEEDED_FILES_FOR_STATISTICS_MD_FILENAME,
    AI_REPRODUCIBILITY_STAMP_FILENAME,
    AI_SETTINGS_SNAPSHOT_JSON_FILENAME,
    AI_SETTINGS_SUMMARY_MD_FILENAME,
)


def _display_generated_path(path: Path, export_dir: Path, deterministic: bool) -> str:
    if not deterministic:
        return str(path)
    return f"<deterministic_export_dir>/{path.relative_to(export_dir).as_posix()}"


def write_hardware_date_recommendation(export_dir: Path) -> Path:
    path = export_dir / AI_HARDWARE_DATE_RECOMMENDATION_MD_FILENAME
    lines = [
        "# Hardware Date Boundary Recommendation",
        "",
        "## Recommended Boundaries",
        "",
        "- RTL-SDR V4 -> Airspy Mini (main analysis):",
        "  - intervention_date: 2026-01-14",
        "  - post_change_date: 2026-01-14",
        "- RTL-SDR V4 -> Airspy Mini (sensitivity analysis):",
        "  - intervention_date: 2026-01-10",
        "  - post_change_date: 2026-01-10",
        "- Airspy-only micro-difference evaluation start:",
        "  - report_start_date: 2026-01-14",
        "- Cable comparison:",
        "  - post_change_date: 2026-02-14",
        "- Adapter comparison:",
        "  - post_change_date: 2026-03-01",
        "- Cable v2 (reference only):",
        "  - post_change_date: 2026-02-26",
        "  - policy: exploratory / reference only (exclude from main conclusion)",
        "",
        "## Statistical Context",
        "",
        "- The large RTL-SDR -> Airspy shift can mask later cable / adapter / parameter micro-effects,",
        "  so Airspy-only evaluation is required.",
        "- Use 2026-01-14 as the main-analysis boundary to represent the hardware transition cleanly.",
        "- Use 2026-01-10 as the sensitivity-analysis boundary to test robustness when transition-period effects are included.",
        "- Start adapter comparisons on 2026-03-01 and treat 2026-02-28 as a mixed day outside the primary conclusion.",
        "- Keep Cable v2 as a reference comparison only, not as evidence for the main conclusion.",
        "",
        "## Practical Use",
        "",
        "- Fix the main analysis at the 2026-01-14 boundary and report the 2026-01-10 sensitivity analysis separately.",
        "- Include an Airspy-only subset in the standard report and evaluate micro-differences there.",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")
    return path


def write_needed_files_for_statistics(export_dir: Path, candidates: list[AICandidateFile]) -> Path:
    path = export_dir / AI_NEEDED_FILES_FOR_STATISTICS_MD_FILENAME
    lines = [
        "# Needed Files For Statistics",
        "",
        "Candidate files required for future statistical decisions.",
        "",
    ]
    for candidate in candidates:
        lines.extend(
            [
                f"## {candidate.logical_name}",
                f"- path: `{candidate.expected_path}`",
                f"- priority: {candidate.priority}",
                f"- reason: {candidate.reason}",
                f"- what_it_enables: {candidate.what_it_enables}",
                "",
            ]
        )
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")
    return path


def write_analysis_design_note(export_dir: Path, hardware_note_path: Path, change_point_note_path: Path) -> Path:
    path = export_dir / AI_ANALYSIS_DESIGN_MD_FILENAME
    hardware_text = hardware_note_path.read_text(encoding="utf-8") if hardware_note_path.exists() else ""
    change_point_text = change_point_note_path.read_text(encoding="utf-8") if change_point_note_path.exists() else ""
    lines = [
        "# Analysis Design",
        "",
        "This document merges the hardware date recommendation and change-point setting notes.",
        "",
        "## Hardware Date Recommendation",
        "",
    ]
    lines.extend(hardware_text.strip().splitlines() if hardware_text.strip() else ["(missing)"])
    lines.extend(["", "## Change Point Settings Note", ""])
    lines.extend(change_point_text.strip().splitlines() if change_point_text.strip() else ["(missing)"])
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")
    return path


def write_analysis_methodology(export_dir: Path) -> Path:
    path = export_dir / AI_ANALYSIS_METHODOLOGY_MD_FILENAME
    lines = [
        "# Analysis Methodology",
        "",
        "## Scope",
        "- Main hardware boundary: 2026-01-14 (RTL-SDR -> Airspy)",
        "- Sensitivity boundary: 2026-01-10",
        "- Airspy-only micro-difference evaluations should start from 2026-01-14",
        "",
        "## Comparative Strategy",
        "- Separate large RTL->Airspy effect from later cable/adapter micro-effects.",
        "- Treat cable v2 as exploratory/reference, not as primary conclusion.",
        "",
        "## Data & Artifacts",
        "- Use daily summary as primary outcome and phase mapping for transition handling.",
        "- Use change-point outputs and phase evaluator outputs as cross-check evidence.",
        "- Use OpenSky and PLAO summaries for traffic-normalized and distance-band interpretation.",
        "",
        "## Quality Notes",
        "- Keep production settings unchanged for primary conclusions.",
        "- Run fine-grained distance bins as comparison experiments due N-per-bin tradeoffs.",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")
    return path


def append_statistical_context_to_summary(
    lines: list[str],
    candidate_statuses: list[AICandidateStatus],
    hardware_note_path: Path,
    needed_files_note_path: Path,
    candidate_status_csv_path: Path,
    extended_manifest_path: Path,
) -> None:
    missing = [status for status in candidate_statuses if not status.exists]
    lines.extend(
        [
            "hardware_date_recommendation_md:",
            f"- {hardware_note_path}",
            "needed_files_for_statistics_md:",
            f"- {needed_files_note_path}",
            "candidate_status_csv:",
            f"- {candidate_status_csv_path}",
            "extended_manifest_csv:",
            f"- {extended_manifest_path}",
            "date_boundary_recommendation_summary:",
            "- RTL-SDR -> Airspy main analysis: intervention_date=2026-01-14, post_change_date=2026-01-14",
            "- RTL-SDR -> Airspy sensitivity analysis: intervention_date=2026-01-10, post_change_date=2026-01-10",
            "- Airspy-only micro-difference evaluation: report_start_date=2026-01-14",
            "- cable comparison: post_change_date=2026-02-14",
            "- adapter comparison: post_change_date=2026-03-01 (2026-02-28 is mixed day)",
            "- cable v2: post_change_date=2026-02-26 (exploratory/reference only)",
            "key_statistical_points:",
            "- RTL->Airspy large shift should be separated from later cable/adapter micro-effects.",
            "- Main conclusion should rely on 2026-01-14 boundary.",
            "- 2026-01-10 is for sensitivity analysis, not the primary boundary.",
            "- Adapter evaluation should start from 2026-03-01 due to mixed day handling.",
            "- Cable v2 should remain reference-only in interpretation.",
            "future_statistics_candidate_files:",
        ]
    )
    for status in candidate_statuses:
        lines.append(
            "- {} | path={} | exists={} | copied_to_export={} | priority={}".format(
                status.logical_name,
                status.expected_path,
                int(status.exists),
                int(status.copied_to_export),
                status.priority,
            )
        )
    lines.append("missing candidate files list:")
    if missing:
        for status in missing:
            lines.append(f"- {status.logical_name}: {status.expected_path}")
    else:
        lines.append("- (none)")


def write_ai_settings_snapshot(export_dir: Path, generated_at: str) -> tuple[Path, Path]:
    settings = load_settings(force_reload=True)
    min_auc_n_used, min_minutes_covered = get_quality_thresholds()
    payload = {
        "generated_at": generated_at,
        "settings_path": settings.path,
        "quality_thresholds": {
            "min_auc_n_used": min_auc_n_used,
            "min_minutes_covered": min_minutes_covered,
        },
        "settings": settings.data,
    }

    json_path = export_dir / AI_SETTINGS_SNAPSHOT_JSON_FILENAME
    with json_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    md_path = export_dir / AI_SETTINGS_SUMMARY_MD_FILENAME
    md_lines = [
        "# AI Settings Summary",
        "",
        f"- generated_at: {generated_at}",
        f"- settings_path: {settings.path}",
        f"- min_auc_n_used (get_quality_thresholds): {min_auc_n_used}",
        f"- min_minutes_covered (get_quality_thresholds): {min_minutes_covered}",
        "",
        "## Change Point Scripts Note",
        "",
        "- `adsb_detect_change_point.py` and `adsb_detect_multi_change_points.py`",
        "- `adsb_detect_change_point.py` and `adsb_detect_multi_change_points.py` call",
        "  `arena.lib.config.get_quality_thresholds()` to resolve quality thresholds.",
        "- `get_quality_thresholds()` reads `quality.min_auc_n_used` and",
        "  `quality.min_minutes_covered` from `settings.toml` (`scripts/config/settings.toml`).",
        "- As a result, change-point quality thresholds are managed through `settings.toml`.",
        "",
        "## settings.toml snapshot",
        "",
        "```json",
        json.dumps(settings.data, ensure_ascii=False, indent=2),
        "```",
    ]
    with md_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(md_lines) + "\n")

    return json_path, md_path


def write_ai_change_point_note(export_dir: Path, settings_path: str) -> Path:
    note_path = export_dir / AI_CHANGE_POINT_NOTE_MD_FILENAME
    lines = [
        "# Change Point Configuration Note",
        "",
        "Configuration references used by the change-point analysis included in this export:",
        "",
        "- Scripts:",
        "  - `scripts/adsb/analysis/change_points/adsb_detect_change_point.py`",
        "  - `scripts/adsb/analysis/change_points/adsb_detect_multi_change_points.py`",
        "- Configuration accessor:",
        "  - `arena.lib.config.get_quality_thresholds()`",
        "- Effective settings file:",
        f"  - `{settings_path}`",
        "",
        "Key parameters:",
        "- `quality.min_auc_n_used`",
        "- `quality.min_minutes_covered`",
        "",
        "Notes:",
        "- The scripts do not read `settings.toml` directly.",
        "- They reference it indirectly through `get_quality_thresholds()`.",
    ]
    with note_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")
    return note_path


def write_ai_export_summary(
    summary_path: Path,
    generated_at: str,
    source_base_dir: Path,
    export_dir: Path,
    records: list[AIManifestRecord],
    settings_json_path: Path,
    settings_md_path: Path,
    change_point_note_path: Path,
    integrity: AIExportIntegrity,
    hardware_note_path: Path,
    needed_files_note_path: Path,
    candidate_status_csv_path: Path,
    extended_manifest_path: Path,
    candidate_statuses: list[AICandidateStatus],
    provenance_path: Path,
    run_metadata_path: Path,
    lineage_path: Path,
    integrity_summary_path: Path,
    artifact_index_path: Path,
    bundle_sha256: str,
    deterministic: bool = False,
) -> None:
    required_records = [record for record in records if record.required_level == "required"]
    recommended_records = [record for record in records if record.required_level == "recommended"]
    required_found = sum(1 for record in required_records if record.exists)
    required_missing = sum(1 for record in required_records if not record.exists)
    recommended_found = sum(1 for record in recommended_records if record.exists)
    recommended_missing = sum(1 for record in recommended_records if not record.exists)
    copied_total = sum(1 for record in records if record.copied)
    missing_records = [record for record in records if record.status in {"missing_required", "missing_recommended", "copy_failed"}]
    duplicate_skipped_records = [record for record in records if record.status == "duplicate_source_skipped"]
    excluded_records = [record for record in records if record.status == "excluded_by_rule"]

    lines = [
        f"generated_at: {generated_at}",
        f"source_base_dir: {source_base_dir}",
        f"export_dir: {'<deterministic_export_dir>' if deterministic else export_dir}",
        f"required_total: {len(required_records)}",
        f"required_found: {required_found}",
        f"required_missing: {required_missing}",
        f"recommended_total: {len(recommended_records)}",
        f"recommended_found: {recommended_found}",
        f"recommended_missing: {recommended_missing}",
        f"copied_total: {copied_total}",
        f"settings_snapshot_json: {_display_generated_path(settings_json_path, export_dir, deterministic)}",
        f"settings_summary_md: {_display_generated_path(settings_md_path, export_dir, deterministic)}",
        f"change_point_note_md: {_display_generated_path(change_point_note_path, export_dir, deterministic)}",
        f"integrity_passed: {int(integrity.passed)}",
        f"integrity_duplicate_source_skipped: {integrity.duplicate_source_skipped}",
        f"integrity_duplicate_output_paths: {integrity.duplicate_output_paths}",
        f"integrity_missing_artifacts: {integrity.missing_artifacts}",
        f"integrity_hash_mismatches: {integrity.hash_mismatches}",
        f"integrity_validated_hash_entries: {integrity.validated_hash_entries}",
        f"excluded_by_rule_total: {len(excluded_records)}",
        f"artifact_hashes_txt: {_display_generated_path(export_dir / AI_ARTIFACT_HASHES_FILENAME, export_dir, deterministic)}",
        f"reproducibility_stamp_json: {_display_generated_path(export_dir / AI_REPRODUCIBILITY_STAMP_FILENAME, export_dir, deterministic)}",
        f"artifact_provenance_json: {_display_generated_path(provenance_path, export_dir, deterministic)}",
        f"run_metadata_json: {_display_generated_path(run_metadata_path, export_dir, deterministic)}",
        f"artifact_lineage_json: {_display_generated_path(lineage_path, export_dir, deterministic)}",
        f"integrity_summary_json: {_display_generated_path(integrity_summary_path, export_dir, deterministic)}",
        f"artifact_index_json: {_display_generated_path(artifact_index_path, export_dir, deterministic)}",
        f"bundle_sha256: {bundle_sha256}",
        "exclude_patterns:",
    ]
    for pattern in AI_EXPORT_EXCLUDE_GLOB_PATTERNS:
        lines.append(f"- {pattern}")

    lines.append("missing files list:")
    if missing_records:
        for record in missing_records:
            lines.append(f"- {record.required_level}: {record.relative_path} ({record.status})")
    else:
        lines.append("- (none)")

    lines.append("duplicate source skipped list:")
    if duplicate_skipped_records:
        for record in duplicate_skipped_records:
            lines.append(f"- {record.required_level}: {record.relative_path} ({record.note})")
    else:
        lines.append("- (none)")

    lines.append("excluded by rule list:")
    if excluded_records:
        for record in excluded_records:
            lines.append(f"- {record.required_level}: {record.relative_path} ({record.status})")
    else:
        lines.append("- (none)")

    append_statistical_context_to_summary(
        lines=lines,
        candidate_statuses=candidate_statuses,
        hardware_note_path=Path(_display_generated_path(hardware_note_path, export_dir, deterministic)),
        needed_files_note_path=Path(_display_generated_path(needed_files_note_path, export_dir, deterministic)),
        candidate_status_csv_path=Path(_display_generated_path(candidate_status_csv_path, export_dir, deterministic)),
        extended_manifest_path=Path(_display_generated_path(extended_manifest_path, export_dir, deterministic)),
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")
