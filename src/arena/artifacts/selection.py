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
            reason="実際の変更時刻と設定変更内容を復元するため。",
            what_it_enables="2026-01-10〜2026-01-14 のズレ解消、mixed day 判定。",
            rationale="hardware_transition_timing",
        ),
        AICandidateFile(
            logical_name="phase_config_daily_mapping",
            relative_path="performance/phase_config_daily_mapping.csv",
            expected_path="output/performance/phase_config_daily_mapping.csv",
            priority="A",
            reason="日次ラベルと phase 遷移日の整合を確認するため。",
            what_it_enables="transition day 除外の再解析。",
            rationale="phase_label_verification",
        ),
        AICandidateFile(
            logical_name="adsb_daily_summary_v2",
            relative_path="adsb_daily_summary_v2.csv",
            expected_path="output/adsb_daily_summary_v2.csv",
            priority="A",
            reason="日次の主要アウトカムを統一フォーマットで確認するため。",
            what_it_enables="再集計、主解析・感度分析の両方の比較。",
            rationale="daily_primary_outcome",
        ),
        AICandidateFile(
            logical_name="adsb_timebin_summary",
            relative_path="time_resolved/adsb_timebin_summary.csv",
            expected_path="output/time_resolved/adsb_timebin_summary.csv",
            priority="B",
            reason="時間帯別の差分を確認するため。",
            what_it_enables="night gain や時間帯依存差の考察。",
            rationale="time_bin_effects",
        ),
        AICandidateFile(
            logical_name="opensky_comparison_daily_summary",
            relative_path="opensky_comparison/opensky_comparison_daily_summary.csv",
            expected_path="output/opensky_comparison/opensky_comparison_daily_summary.csv",
            priority="B",
            reason="トラフィックを相対化して比較するため。",
            what_it_enables="capture 効率比較。",
            rationale="traffic_normalization",
        ),
        AICandidateFile(
            logical_name="plao_daily_distance_auc_summary",
            relative_path="plao/distance_auc/plao_daily_distance_auc_summary.csv",
            expected_path="output/plao/distance_auc/plao_daily_distance_auc_summary.csv",
            priority="B",
            reason="距離帯別の差分を確認するため。",
            what_it_enables="near/mid/far の差の考察。",
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
