from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from arena.lib.paths import OUTPUT_DIR as ARENA_OUTPUT_DIR

ROOT = Path(__file__).resolve().parents[3]


OUTPUT_DIR = ARENA_OUTPUT_DIR


TEXT_EXT_DEFAULT = [".txt", ".log", ".json", ".jsonl", ".csv", ".html", ".md"]
ALWAYS_EXCLUDE_DIRS = {"merged_for_ai", "_tmp_check_out"}
ALWAYS_EXCLUDE_DIR_PREFIXES = ("_tmp",)
ALWAYS_EXCLUDE_REL_PATHS = {"performance/pipeline_runs.jsonl"}
NORMAL_MODE_OPTIONAL_EXPORT_FILES = [
    "adsb_daily_summary_v2.csv",
    "performance/phase_config_daily_mapping.csv",
    "performance/pipeline_runs.jsonl",
]

AI_EXPORT_DIR_PREFIX = "_ai_review_"
AI_MANIFEST_FILENAME = "ai_selected_manifest.csv"
AI_MANIFEST_EXTENDED_FILENAME = "ai_selected_manifest_extended.csv"
AI_SUMMARY_FILENAME = "ai_export_summary.txt"
AI_SETTINGS_SUMMARY_MD_FILENAME = "ai_settings_summary.md"
AI_SETTINGS_SNAPSHOT_JSON_FILENAME = "ai_settings_snapshot.json"
AI_CHANGE_POINT_NOTE_MD_FILENAME = "ai_change_point_settings_note.md"
AI_HARDWARE_DATE_RECOMMENDATION_MD_FILENAME = "ai_hardware_date_recommendation.md"
AI_NEEDED_FILES_FOR_STATISTICS_MD_FILENAME = "ai_needed_files_for_statistics.md"
AI_CANDIDATE_STATUS_CSV_FILENAME = "ai_export_candidate_status.csv"
AI_ANALYSIS_DESIGN_MD_FILENAME = "analysis_design.md"
AI_ANALYSIS_METHODOLOGY_MD_FILENAME = "analysis_methodology.md"
AI_FILES_SUBDIR = "files"
AI_ARTIFACT_HASHES_FILENAME = "artifact_hashes.txt"
AI_REPRODUCIBILITY_STAMP_FILENAME = "reproducibility_stamp.json"
AI_ARTIFACT_PROVENANCE_FILENAME = "artifact_provenance.json"
AI_RUN_METADATA_FILENAME = "run_metadata.json"
AI_ARTIFACT_LINEAGE_FILENAME = "artifact_lineage.json"
AI_ARTIFACT_INDEX_FILENAME = "artifact_index.json"
AI_INTEGRITY_SUMMARY_JSON_FILENAME = "integrity_summary.json"
AI_PACK_DIR_GEMINI = "for_gemini"
AI_PACK_DIR_GPT = "for_GPT"
AI_PACK_DIR_GROK = "for_grok"
AI_PACK_DETAILS_SUBDIR = "details"
AI_GEMINI_CORE_LIMIT = 5
AI_GPT_CORE_LIMIT = 20
ARTIFACT_SUBSYSTEM_VERSION = "1.1.0"
POLICY_VERSION = "1.0"
DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00"
AI_EXPORT_EXCLUDE_GLOB_PATTERNS = [
    "*dist_1m*.jsonl",
    "*dist_1m*.jsonl.*",
    "*dist_1m*minute_raw*",
    "*distance*minute_raw*",
    "*dist*minute_raw*",
    "*dist*raw*",
    "*distance*raw*",
    "*dist*full*",
    "*distance*full*",
    "*raw/past_log/*dist*.jsonl*",
    "*minutely_merged*",
]
AI_PRIORITY_B_ALLOWED_EXTS = {".csv", ".tsv", ".json", ".txt", ".md"}
AI_PRIORITY_B_MAX_FILE_BYTES = 25_000_000
AI_PRIORITY_B_DISCOVERY_INCLUDE_GLOBS = [
    "*power*analysis*.csv",
    "*power*analysis*.json",
    "*power*analysis*.txt",
    "*power*analysis*.md",
    "*mde*.csv",
    "*mde*.json",
    "*mde*.txt",
    "*mde*.md",
    "*detectable*effect*.csv",
    "*detectable*effect*.json",
    "*detectable*effect*.txt",
    "*sample*size*.csv",
    "*sample*size*.json",
    "*sample*size*.txt",
    "*required*n*.csv",
    "*required*n*.json",
    "*required*n*.txt",
    "*sensitivity*analysis*.csv",
    "*sensitivity*analysis*.json",
    "*sensitivity*analysis*.txt",
    "*detectable*delta*.csv",
    "*detectable*delta*.json",
    "*detectable*delta*.txt",
    "*time_bin*summary*.csv",
    "*timebin*summary*.csv",
    "*time_bin*detailed*stats*.csv",
    "*time*resolved*summary*.csv",
    "*phase*time*bin*.csv",
    "*phase*time*bin*summary*.json",
    "*phase*time*bin*export*report*.txt",
    "*phase*config*daily*mapping*.csv",
    "*phase*config*daily*mapping*.json",
    "*by_phase*time*.csv",
    "*hourly*summary*.csv",
    "*diurnal*summary*.csv",
    "*distance*summary*.csv",
    "*distance*report*.txt",
    "*distance*result*.json",
    "*distance*stats*report*.txt",
    "*distance*auc*summary*.csv",
    "*distance*auc*long*.csv",
    "*distance*auc*stats*report*.txt",
    "*distance*binomial*summary*.csv",
    "*distance*performance*summary*.csv",
    "*quality*binomial*summary*.txt",
    "*phase*config*.toml",
    "*phase*definition*.txt",
    "*phase*definition*.md",
    "*phase*baseline*.txt",
    "*phase*baseline*.md",
    "*settings*snapshot*.json",
    "*settings*summary*.md",
    "*opensky*comparison*daily*summary*.csv",
    "*opensky*comparison*stats*report*.txt",
    "*opensky*skipped*days*.csv",
]
AI_PRIORITY_B_DISCOVERY_EXCLUDE_GLOBS = [
    "*cache*",
    "*tmp*",
    "*temp*",
    "*notebook*",
    "*ipynb*",
    "*debug*",
    "*plot*",
    "*figure*",
    "*pipeline_runs_*test*.jsonl",
    "*raw*",
    "*full*",
    "*minute_raw*",
    "*minutely*",
]
AI_REQUIRED_FILES = [
    "change_point/change_point_report.txt",
    "change_point/multi_change_points_report.txt",
    "adsb_daily_summary_v2.csv",
    "phase_evaluator_report.txt",
    "coverage/coverage_trend.csv",
    "performance/distance_binomial_summary.csv",
    "fringe_decoding/fringe_decoding_stats.csv",
    "fringe_decoding/fringe_decoding_trend.csv",
    "adsb_signal_daily_summary.csv",
    "adsb_signal_range_summary.csv",
    "performance/phase_evaluator_results.csv",
    "performance/phase_evaluator_report.txt",
    "performance/bayesian_phase_results_cuda.csv",
    "opensky_comparison/opensky_comparison_daily_summary.csv",
    "phases_v3_baseline.txt",
    "vertical_profile/los_efficiency_trend.csv",
    "time_resolved/adsb_timebin_summary.csv",
]
AI_RECOMMENDED_FILES = [
    "change_point/change_point_result.json",
    "change_point/multi_change_points_result.json",
    "change_point/change_point_histogram.png",
    "change_point/multi_change_points_histogram.png",
    "opensky_phase_comparison_report.txt",
    "opensky_phase_comparison.csv",
    "opensky_comparison/opensky_skipped_days.csv",
    "bayesian_phase_results.csv",
    "adsb_daily_summary_raw.csv",
    "performance/time_bin_detailed_stats.csv",
    "performance/phase_config_daily_mapping.csv",
    "performance/phase_config_daily_mapping.json",
    "performance/phase_timebin_summary.csv",
    "performance/phase_timebin_summary.json",
    "performance/phase_timebin_export_report.txt",
    "performance/baseline_nb_summary.txt",
    "performance/baseline_nb_results.json",
    "performance/distance_performance_summary.csv",
    "performance/quality_binomial_summary.txt",
    "performance/pipeline_runs.jsonl",
    "opensky_comparison/opensky_comparison_stats_report.txt",
    "plao/distance_auc/plao_daily_distance_auc_summary.csv",
    "plao/distance_auc/plao_daily_distance_auc_long.csv",
    "plao/distance_auc/plao_distance_auc_stats_report.txt",
    "plao/distance_auc/plao_skipped_days.csv",
    "settings.toml",
    "scripts/structure",
]
AI_FALLBACK_SOURCE_PATHS = {
    "phases_v3_baseline.txt": ROOT / "scripts" / "config" / "phases_v3_airspy_baseline.txt",
    "settings.toml": ROOT / "scripts" / "config" / "settings.toml",
    "scripts/structure": ROOT / "scripts" / "structure",
}
AI_FILENAME_GLOB_FALLBACKS = {
    "change_point/change_point_report.txt": [
        "*change*point*report*.txt",
        "*change*point*summary*.txt",
        "*change_point*.txt",
    ],
    "change_point/multi_change_points_report.txt": [
        "*multi*change*point*report*.txt",
        "*multi*change*point*summary*.txt",
        "*multi*change*point*.txt",
    ],
    "change_point_report.txt": [
        "*change*point*report*.txt",
        "*change*point*summary*.txt",
        "*change_point*.txt",
    ],
    "multi_change_points_report.txt": [
        "*multi*change*point*report*.txt",
        "*multi*change*point*summary*.txt",
        "*multi*change*point*.txt",
    ],
    "opensky_phase_comparison.csv": [
        "*opensky*phase*comparison*.csv",
        "opensky_comparison_daily_summary.csv",
        "*opensky*comparison*daily*summary*.csv",
    ],
    "opensky_phase_comparison_report.txt": [
        "*opensky*phase*comparison*report*.txt",
        "opensky_comparison_stats_report.txt",
        "*opensky*comparison*stats*report*.txt",
    ],
    "opensky_comparison/opensky_skipped_days.csv": [
        "opensky_skipped_days.csv",
        "*opensky*skipped*days*.csv",
        "*opensky*skip*days*.csv",
    ],
    "time_resolved/adsb_timebin_summary.csv": [
        "adsb_timebin_summary.csv",
        "*time*bin*summary*.csv",
        "*timebin*summary*.csv",
    ],
    "performance/time_bin_detailed_stats.csv": [
        "time_bin_detailed_stats.csv",
        "*time*bin*detailed*stats*.csv",
    ],
    "plao/distance_auc/plao_daily_distance_auc_long.csv": [
        "plao_daily_distance_auc_long.csv",
        "*distance*auc*long*.csv",
        "*distance*minute*long*.csv",
        "*dist*1m*long*.csv",
    ],
    "performance/pipeline_runs.jsonl": [
        "pipeline_runs.jsonl",
        "*pipeline*run*.jsonl",
        "*pipeline_runs*.jsonl",
    ],
    "performance/quality_binomial_summary.txt": [
        "quality_binomial_summary.txt",
        "*quality*binomial*summary*.txt",
    ],
    "performance/phase_config_daily_mapping.csv": [
        "phase_config_daily_mapping.csv",
        "*phase*config*daily*mapping*.csv",
    ],
    "performance/phase_config_daily_mapping.json": [
        "phase_config_daily_mapping.json",
        "*phase*config*daily*mapping*.json",
    ],
    "performance/phase_timebin_summary.csv": [
        "phase_timebin_summary.csv",
        "*phase*time*bin*summary*.csv",
    ],
    "performance/phase_timebin_summary.json": [
        "phase_timebin_summary.json",
        "*phase*time*bin*summary*.json",
    ],
    "performance/phase_timebin_export_report.txt": [
        "phase_timebin_export_report.txt",
        "*phase*time*bin*export*report*.txt",
    ],
    "plao/distance_auc/plao_skipped_days.csv": [
        "plao_skipped_days.csv",
        "*distance*auc*skipped*days*.csv",
    ],
}
AI_GEMINI_PACK_KEYS = [
    AI_SUMMARY_FILENAME,
    AI_HARDWARE_DATE_RECOMMENDATION_MD_FILENAME,
    AI_NEEDED_FILES_FOR_STATISTICS_MD_FILENAME,
    "adsb_daily_summary_v2.csv",
    "phase_config_daily_mapping.csv",
]
AI_GPT_PACK_KEYS = [
    "plao_daily_distance_auc_summary.csv",
    "plao_distance_auc_stats_report.txt",
    "opensky_phase_comparison_report.txt",
    "opensky_comparison_daily_summary.csv",
    "phase_config_daily_mapping.csv",
    "phase_config_daily_mapping.json",
    "phases_v3_baseline.txt",
    "adsb_daily_summary_v2.csv",
    "adsb_timebin_summary.csv",
    "phase_timebin_summary.csv",
    "phase_timebin_summary.json",
    "multi_change_points_report.txt",
    "multi_change_points_result.json",
    "baseline_nb_summary.txt",
    "baseline_nb_results.json",
    AI_ANALYSIS_METHODOLOGY_MD_FILENAME,
    "phase_evaluator_report.txt",
    "phase_evaluator_results.csv",
    "change_point_report.txt",
    "change_point_result.json",
]
CATEGORY_MAP = {
    "change_point/change_point_report.txt": "change_points",
    "change_point/multi_change_points_report.txt": "change_points",
    "change_point/change_point_result.json": "change_points",
    "change_point/multi_change_points_result.json": "change_points",
    "change_point/change_point_histogram.png": "change_points",
    "change_point/multi_change_points_histogram.png": "change_points",
    "change_point_report.txt": "change_points",
    "multi_change_points_report.txt": "change_points",
    "change_point_result.json": "change_points",
    "multi_change_points_result.json": "change_points",
    "change_point_histogram.png": "change_points",
    "multi_change_points_histogram.png": "change_points",
    "adsb_daily_summary_v2.csv": "adsb",
    "adsb_daily_summary_raw.csv": "adsb",
    "coverage/coverage_trend.csv": "coverage",
    "performance/distance_binomial_summary.csv": "performance",
    "distance_binomial_summary.csv": "performance",
    "distance_performance_summary.csv": "performance",
    "fringe_decoding/fringe_decoding_stats.csv": "fringe_decoding",
    "fringe_decoding_stats.csv": "fringe_decoding",
    "fringe_decoding/fringe_decoding_trend.csv": "fringe_decoding",
    "adsb_signal_daily_summary.csv": "signal",
    "adsb_signal_range_summary.csv": "signal",
    "performance/phase_evaluator_results.csv": "performance",
    "performance/phase_evaluator_report.txt": "performance",
    "phase_evaluator_report.txt": "performance",
    "performance/bayesian_phase_results_cuda.csv": "performance",
    "bayesian_phase_results.csv": "performance",
    "opensky_phase_comparison_report.txt": "opensky_comparison",
    "opensky_phase_comparison.csv": "opensky_comparison",
    "opensky_comparison/opensky_local_minutely_merged.csv": "opensky_comparison",
    "opensky_comparison/opensky_skipped_days.csv": "opensky_comparison",
    "opensky_comparison/opensky_comparison_daily_summary.csv": "opensky_comparison",
    "phases_v3_baseline.txt": "config",
    "settings.toml": "config",
    "vertical_profile/los_efficiency_trend.csv": "vertical_profile",
    "time_resolved/adsb_timebin_summary.csv": "time_resolved",
    "performance/time_bin_detailed_stats.csv": "performance",
    "performance/phase_config_daily_mapping.csv": "phase_config",
    "performance/phase_config_daily_mapping.json": "phase_config",
    "performance/phase_timebin_summary.csv": "time_resolved",
    "performance/phase_timebin_summary.json": "time_resolved",
    "performance/phase_timebin_export_report.txt": "time_resolved",
    "performance/baseline_nb_summary.txt": "performance",
    "performance/baseline_nb_results.json": "performance",
    "performance/distance_performance_summary.csv": "performance",
    "performance/quality_binomial_summary.txt": "performance",
    "opensky_comparison/opensky_comparison_stats_report.txt": "opensky_comparison",
    "plao/distance_auc/plao_daily_distance_auc_summary.csv": "plao_distance_auc",
    "plao/distance_auc/plao_daily_distance_auc_long.csv": "plao_distance_auc",
    "plao/distance_auc/plao_distance_auc_stats_report.txt": "plao_distance_auc",
    "plao/distance_auc/plao_skipped_days.csv": "plao_distance_auc",
    "performance/pipeline_runs.jsonl": "pipeline",
    "scripts/structure": "structure",
}


def infer_category(relative_path: str) -> str:
    return CATEGORY_MAP.get(relative_path, "uncategorized")


def check_ai_export_exclusion(relative_path: str, source_path: Path) -> tuple[bool, str]:
    rel = relative_path.replace("\\", "/").lower()
    src = source_path.as_posix().lower()
    for pattern in AI_EXPORT_EXCLUDE_GLOB_PATTERNS:
        if fnmatch(rel, pattern) or fnmatch(src, pattern):
            return True, f"matched_exclude_pattern={pattern}"

    has_distance = any(key in rel or key in src for key in ["dist_1m", "distance"])
    has_raw_like = any(key in rel or key in src for key in ["minute_raw", "minutely", "/raw/", "_raw", "full"])
    if has_distance and has_raw_like:
        return True, "matched_exclude_rule=distance_raw_or_full_dataset"

    return False, ""

