from __future__ import annotations

from pathlib import Path

from arena.artifacts.discovery import (
    discover_latest_change_points_targets,
    discover_priority_b_existing_targets,
)
from arena.artifacts.models import AICandidateFile
from arena.artifacts.policies import AI_RECOMMENDED_FILES, AI_REQUIRED_FILES


def get_ai_candidate_files() -> list[AICandidateFile]:
    return [
        AICandidateFile(
            logical_name="pipeline_runs",
            relative_path="performance/pipeline_runs.jsonl",
            expected_path="output/performance/pipeline_runs.jsonl",
            priority="A",
            reason="To restore the actual modification time and configuration changes.",
            what_it_enables="Resolve the gap between 2026-01-10 and 2026-01-14, mixed day determination.",
            rationale="hardware_transition_timing",
        ),
        AICandidateFile(
            logical_name="phase_config_daily_mapping",
            relative_path="performance/phase_config_daily_mapping.csv",
            expected_path="output/performance/phase_config_daily_mapping.csv",
            priority="A",
            reason="To check the consistency of daily labels and phase transition dates.",
            what_it_enables="Reanalysis of transition day exclusions.",
            rationale="phase_label_verification",
        ),
        AICandidateFile(
            logical_name="adsb_daily_summary_v2",
            relative_path="adsb_daily_summary_v2.csv",
            expected_path="output/adsb_daily_summary_v2.csv",
            priority="A",
            reason="To confirm daily key outcomes in a unified format.",
            what_it_enables="Re-aggregation, comparison of both main analysis and sensitivity analysis.",
            rationale="daily_primary_outcome",
        ),
        AICandidateFile(
            logical_name="adsb_timebin_summary",
            relative_path="time_resolved/adsb_timebin_summary.csv",
            expected_path="output/time_resolved/adsb_timebin_summary.csv",
            priority="B",
            reason="To check the differences by time zone.",
            what_it_enables="Consideration of night gain and time-zone dependent differences.",
            rationale="time_bin_effects",
        ),
        AICandidateFile(
            logical_name="opensky_comparison_daily_summary",
            relative_path="opensky_comparison/opensky_comparison_daily_summary.csv",
            expected_path="output/opensky_comparison/opensky_comparison_daily_summary.csv",
            priority="B",
            reason="To relativize and compare traffic.",
            what_it_enables="capture efficiency comparison.",
            rationale="traffic_normalization",
        ),
        AICandidateFile(
            logical_name="plao_daily_distance_auc_summary",
            relative_path="plao/distance_auc/plao_daily_distance_auc_summary.csv",
            expected_path="output/plao/distance_auc/plao_daily_distance_auc_summary.csv",
            priority="B",
            reason="To check the difference by distance zone.",
            what_it_enables="Consideration of near/mid/far differences.",
            rationale="distance_band_effects",
        ),
    ]


def candidate_map_by_relative_path() -> dict[str, AICandidateFile]:
    return {candidate.relative_path: candidate for candidate in get_ai_candidate_files()}


def iter_ai_targets(base_dir: Path) -> list[tuple[str, str]]:
    selected: dict[str, str] = {}
    for path in AI_REQUIRED_FILES:
        if path not in selected:
            selected[path] = "required"
    for path in AI_RECOMMENDED_FILES:
        if path not in selected:
            selected[path] = "recommended"
    for candidate in get_ai_candidate_files():
        if candidate.relative_path not in selected:
            selected[candidate.relative_path] = "recommended"
    for path in discover_latest_change_points_targets(base_dir):
        if path not in selected:
            selected[path] = "recommended"
    for path in discover_priority_b_existing_targets(base_dir):
        if path not in selected:
            selected[path] = "recommended"
    return list(selected.items())
