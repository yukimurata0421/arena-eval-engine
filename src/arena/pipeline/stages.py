from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class Step:
    stage: int
    script_rel: str
    label: str
    timeout_s: int = 600
    critical: bool = False
    est_s: int = 0
    input_text: str | None = None
    env_overrides: dict[str, str] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    depends_on_outputs: list[str] = field(default_factory=list)
    always_run_when_skip_existing: bool = False
    soft_fail_on_error: bool = False
    soft_fail_accept_stale_outputs: bool = False
    error_code_base: str = ""
    expected_min_bytes: int = 50
    extra_args: list[str] = field(default_factory=list)
    skip_if_no_inputs: bool = False
    input_dir: str = ""
    input_pattern: str = ""
    input_dir_arg: str = "--input-dir"
    input_pattern_arg: str = "--pattern"


@dataclass
class RunConfig:
    stage: int = 1
    only: int | None = None
    dry_run: bool = False
    no_gpu: bool = False
    full: bool = False
    backend: str = "auto"
    scripts_root: str = ""
    output_root: str = ""
    data_root: str = ""
    dynamic_date: str = ""
    phase_config: str = ""
    validate: bool = True
    validate_only: bool = False
    skip_existing: bool = False
    fail_fast: bool = False
    log_jsonl: str = ""
    log_jsonl_mode: str = "append"
    skip_plao: bool = False
    workers: int = 0


@dataclass
class RunRecord:
    ts_start: str
    ts_end: str
    backend: str
    stage: int
    label: str
    script_rel: str
    status: str
    elapsed_s: float
    returncode: int | None
    cmd: list[str]
    expected_outputs: list[str]
    outputs_ok: bool
    missing_outputs: list[str]
    step_code: str = ""
    error_code: str = ""
    stderr_tail: str = ""
    stdout_tail: str = ""


# =========================
# Validation
# =========================
def _resolve_expected_path(output_root_native: Path, p: str) -> Path:
    if p.startswith("data://"):
        # Logical path under data root (resolved by validate_outputs caller).
        return Path(p)
    if p.startswith("output://"):
        p = p[len("output://") :]
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return output_root_native / p


def validate_outputs(
    output_root_native: Path,
    expected_outputs: Sequence[str],
    min_bytes: int,
    min_mtime: float | None = None,
    data_root_native: Path | None = None,
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for expected in expected_outputs:
        if expected.startswith("data://"):
            rel = expected[len("data://") :].lstrip("/\\")
            base = data_root_native or output_root_native
            fp = base / rel
        else:
            fp = _resolve_expected_path(output_root_native, expected)
        if not fp.exists():
            missing.append(str(fp))
            continue
        try:
            if fp.is_dir():
                files = [p for p in fp.rglob("*") if p.is_file()]
                if not files:
                    missing.append(f"{fp} (empty dir)")
                    continue
                if min_mtime is not None:
                    latest = max(p.stat().st_mtime for p in files)
                    if latest < min_mtime:
                        missing.append(
                            f"{fp} (stale: latest mtime {latest:.0f} < {min_mtime:.0f})"
                        )
                continue

            if fp.is_file() and fp.stat().st_size < min_bytes:
                missing.append(f"{fp} (too small: {fp.stat().st_size} bytes)")
                continue
            if min_mtime is not None and fp.is_file():
                if fp.stat().st_mtime < min_mtime:
                    missing.append(
                        f"{fp} (stale: mtime {fp.stat().st_mtime:.0f} < {min_mtime:.0f})"
                    )
        except Exception:
            missing.append(f"{fp} (stat failed)")
    return (len(missing) == 0), missing

STAGE_NAMES = {
    1: "Aggregation",
    2: "Spatial / Visualization",
    3: "Statistics (CPU)",
    4: "Phase Evaluation",
    5: "Bayesian / Change Points",
    6: "Final Reports",
    7: "PLAO",
    8: "OpenSky Comparison",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_posint(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return val if val > 0 else default


def build_pipeline(dynamic_date: str, full_mode: bool, skip_plao: bool = False) -> list[Step]:
    """
    Define steps with expected outputs (key artifacts only).

    Stage layout (v4):
      1: Aggregation
      2: Spatial / Visual
      3: Stats (CPU) — statsmodels, scipy
      4: Phase Eval  — NumPyro NUTS (CPU; adsb_phase_evaluator.py)
      5: Bayesian / Change Points — PyMC, NumPyro DiscreteHMCGibbs
      6: Final Reports
      7: PLAO — independent data source (plao_distance_auc_eval.py)
    """
    steps: list[Step] = []
    stage_seq: dict[int, int] = {}

    def add(**kwargs) -> None:
        stage = int(kwargs["stage"])
        stage_seq[stage] = stage_seq.get(stage, 0) + 1
        kwargs.setdefault("error_code_base", f"S{stage}-{stage_seq[stage]:02d}")
        steps.append(Step(**kwargs))

    # ================================================================
    # Stage 1: Aggregation
    # ================================================================
    add(
        stage=1,
        script_rel="adsb/aggregators/adsb_aggregator.py",
        label="Daily AUC aggregation (raw -> adsb_daily_summary_raw.csv)",
        error_code_base="S1-01",
        timeout_s=300,
        critical=True,
        est_s=60,
        always_run_when_skip_existing=True,
        expected_outputs=["adsb_daily_summary_raw.csv"],
        expected_min_bytes=200,
    )
    add(
        stage=1,
        script_rel="adsb/aggregators/adsb_eval_pk_aggregator.py",
        label="Daily summary merge (-> adsb_daily_summary.csv)",
        error_code_base="S1-02",
        timeout_s=300,
        critical=True,
        est_s=120,
        always_run_when_skip_existing=True,
        expected_outputs=["adsb_daily_summary.csv"],
        expected_min_bytes=500,
    )
    add(
        stage=1,
        script_rel="adsb/aggregators/adsb_csv_patcher.py",
        label="CSV column patch (hardware/is_post_change, etc.)",
        error_code_base="S1-04",
        timeout_s=120,
        critical=True,
        est_s=10,
        always_run_when_skip_existing=True,
        expected_outputs=["adsb_daily_summary.csv"],
        expected_min_bytes=500,
    )
    add(
        stage=1,
        script_rel="signals/collectors/signal_stats_aggregator.py",
        label="Signal strength aggregation (-> adsb_signal_range_summary.csv)",
        error_code_base="S1-05",
        timeout_s=300,
        est_s=60,
        always_run_when_skip_existing=True,
        expected_outputs=["adsb_signal_range_summary.csv"],
        expected_min_bytes=20,
    )
    add(
        stage=1,
        script_rel="adsb/data_fetch/get_opensky_traffic.py",
        label="OpenSky traffic fetch (-> data/flight_data/airport_movements.csv)",
        error_code_base="S1-06",
        timeout_s=300,
        est_s=30,
        always_run_when_skip_existing=True,
        soft_fail_on_error=True,
        soft_fail_accept_stale_outputs=True,
        expected_outputs=["data://flight_data/airport_movements.csv"],
        expected_min_bytes=200,
    )
    add(
        stage=1,
        script_rel="adsb/aggregators/adsb_local_traffic_proxy_gen.py",
        label="local_traffic_proxy generation (-> adsb_daily_summary_v2.csv)",
        error_code_base="S1-07",
        timeout_s=900,
        critical=True,
        est_s=300,
        always_run_when_skip_existing=True,
        expected_outputs=["adsb_daily_summary_v2.csv"],
        expected_min_bytes=500,
    )
    add(
        stage=1,
        script_rel="adsb/analysis/time_resolved/adsb_time_resolved_aggregator.py",
        label="Time-bin AUC aggregation (-> time_resolved/adsb_timebin_summary.csv)",
        error_code_base="S1-08",
        timeout_s=300,
        est_s=120,
        always_run_when_skip_existing=True,
        expected_outputs=["time_resolved/adsb_timebin_summary.csv"],
        expected_min_bytes=300,
    )

    # ================================================================
    # Stage 2: Spatial / Visual
    # ================================================================
    add(
        stage=2,
        script_rel="adsb/analysis/reports/adsb_fringe_decoding_evaluator.py",
        label="Fringe decoding rate (-> fringe_decoding/*)",
        timeout_s=600,
        est_s=120,
        expected_outputs=[
            "fringe_decoding/fringe_decoding_stats.csv",
        ],
        expected_min_bytes=200,
    )
    add(
        stage=2,
        script_rel="adsb/analysis/reports/adsb_polar_coverage_evaluator.py",
        label="Polar coverage (-> coverage/*)",
        timeout_s=600,
        est_s=180,
        expected_outputs=[
            "coverage/coverage_trend.csv",
            "coverage/coverage_trend_report.png",
        ],
        expected_min_bytes=200,
    )
    add(
        stage=2,
        script_rel="adsb/analysis/reports/adsb_vertical_profile_evaluator.py",
        label="LOS efficiency trend (-> vertical_profile/*)",
        timeout_s=600,
        est_s=180,
        expected_outputs=[
            "vertical_profile/los_efficiency_trend.csv",
            "vertical_profile/los_efficiency_trend_report.png",
        ],
        expected_min_bytes=200,
    )
    add(
        stage=2,
        script_rel="adsb/heatmap/adsb_heatmap_generator.py",
        label="Unified heatmap (-> adsb_coverage_heatmap.html)",
        timeout_s=300,
        est_s=30,
        expected_outputs=["adsb_coverage_heatmap.html"],
        expected_min_bytes=200,
    )
    add(
        stage=2,
        script_rel="adsb/heatmap/adsb_daily_heatmap_alt.py",
        label="Altitude-band heatmaps (-> heatmaps/)",
        timeout_s=900,
        est_s=300,
        expected_outputs=[],
    )
    if full_mode:
        add(
            stage=2,
            script_rel="adsb/heatmap/adsb_daily_heatmap.py",
            label="Daily heatmaps (basic)",
            timeout_s=900,
            est_s=300,
            expected_outputs=[],
        )

    # ================================================================
    # Stage 3: Stats (CPU)
    # ================================================================
    add(
        stage=3,
        script_rel="adsb/analysis/stats/adsb_baseline_nb_eval.py",
        label="NB GLM baseline (-> performance/baseline_nb_*)",
        timeout_s=180,
        est_s=20,
        expected_outputs=[
            "performance/baseline_nb_summary.txt",
            "performance/baseline_nb_results.json",
        ],
        expected_min_bytes=150,
    )
    add(
        stage=3,
        script_rel="adsb/analysis/stats/adsb_stats_eval.py",
        label="Statistical evaluation (statsmodels; console)",
        timeout_s=180,
        est_s=20,
        expected_outputs=[],
    )
    add(
        stage=3,
        script_rel="adsb/analysis/stats/adsb_dynamic_eval.py",
        label=f"Dynamic evaluation (intervention date: {dynamic_date})",
        timeout_s=180,
        est_s=20,
        input_text=f"{dynamic_date}\n",
        expected_outputs=[],
    )
    add(
        stage=3,
        script_rel="adsb/analysis/stats/adsb_distance_nb_eval.py",
        label="Distance-band comparison (MWU + bootstrap) (-> performance/distance_performance_summary.csv)",
        timeout_s=240,
        est_s=30,
        expected_outputs=["performance/distance_performance_summary.csv"],
        expected_min_bytes=200,
    )
    add(
        stage=3,
        script_rel="adsb/analysis/stats/adsb_distance_binomial_eval.py",
        label="Distance-band comparison (binomial) (-> performance/distance_binomial_summary.csv)",
        timeout_s=240,
        est_s=20,
        expected_outputs=["performance/distance_binomial_summary.csv"],
        expected_min_bytes=200,
    )
    add(
        stage=3,
        script_rel="adsb/analysis/reports/adsb_fringe_decoding_quality_stats.py",
        label="Fringe decoding quality (v2; console)",
        timeout_s=180,
        est_s=15,
        expected_outputs=[],
    )
    add(
        stage=3,
        script_rel="adsb/analysis/time_resolved/adsb_time_resolved_mixed_eval.py",
        label="Time-bin mixed model (-> performance/time_resolved_performance.png)",
        timeout_s=180,
        est_s=20,
        expected_outputs=["performance/time_resolved_performance.png"],
        expected_min_bytes=200,
    )
    add(
        stage=3,
        script_rel="adsb/analysis/time_resolved/adsb_time_resolved_detailed_report.py",
        label="Time-bin detailed report (-> performance/time_bin_detailed_stats.csv, etc.)",
        timeout_s=180,
        est_s=20,
        expected_outputs=[
            "performance/time_bin_detailed_stats.csv",
            "performance/time_resolved_detailed_plot.png",
        ],
        expected_min_bytes=200,
    )
    add(
        stage=3,
        script_rel="adsb/analysis/phase/adsb_phase_timebin_export.py",
        label="Daily phase/config mapping + phase time-bin aggregation (-> performance/phase_*)",
        timeout_s=180,
        est_s=20,
        expected_outputs=[
            "performance/phase_config_daily_mapping.csv",
            "performance/phase_timebin_summary.csv",
        ],
        expected_min_bytes=120,
    )
    add(
        stage=3,
        script_rel="signals/analysis/signal_range_evaluator.py",
        label="Signal strength by distance band (console)",
        timeout_s=120,
        est_s=10,
        expected_outputs=[],
    )
    add(
        stage=3,
        script_rel="signals/analysis/signal_quality_evaluator.py",
        label="Signal quality for the 150-175 km band (console)",
        timeout_s=120,
        est_s=10,
        expected_outputs=[],
    )

    # ================================================================
    # Stage 4: Phase Evaluation (NumPyro NUTS)
    # ================================================================
    # adsb_phase_evaluator.py: Bayesian phase comparison
    # Compare against Stage 3 (statsmodels) results as a Bayesian view.
    # NumPyro NUTS (continuous params only) is fast enough on CPU.
    add(
        stage=4,
        script_rel="adsb/analysis/phase/adsb_phase_evaluator_v3.py",
        label="Phase Bayesian comparison (dual baseline) (-> performance/phase_evaluator_*)",
        timeout_s=300,
        est_s=30,
        env_overrides={
            "ADSB_BATCH_MODE": "1",
            "ADSB_PHASE_INTERACTIVE": "0",
            "JAX_PLATFORMS": "cpu",
            # Default keeps production-quality sampling, but allow explicit env override for fast runs.
            "ADSB_PHASE_WARMUP": str(_env_posint("ADSB_PHASE_WARMUP", 1000)),
            "ADSB_PHASE_SAMPLES": str(_env_posint("ADSB_PHASE_SAMPLES", 2000)),
        },
        expected_outputs=[
            "performance/phase_evaluator_results.csv",
            "performance/phase_evaluator_report.txt",
            "performance/phase_evaluator_boxplot.png",
        ],
        depends_on_outputs=["adsb_daily_summary.csv"],
        expected_min_bytes=100,
    )

    # ================================================================
    # Stage 5: Bayesian & Change Points
    # ================================================================
    # [v3] Speedup: force CPU for all DiscreteHMCGibbs scripts.
    # At n=59, GPU (GTX 1060) is 38-50x slower than CPU.
    # Measured: Bayesian phase 658s→17s, multi change points 748s→15s.
    add(
        stage=5,
        script_rel="adsb/analysis/gpu/adsb_bayesian_phase_cuda_eval.py",
        label="Bayesian phase comparison (-> performance/bayesian_phase_results_cuda.csv)",
        timeout_s=300,
        est_s=20,
        env_overrides={"JAX_PLATFORMS": "cpu"},
        expected_outputs=["performance/bayesian_phase_results_cuda.csv"],
        expected_min_bytes=200,
    )
    add(
        stage=5,
        script_rel="adsb/analysis/bayesian/adsb_bayesian_dynamic_eval.py",
        label=f"Bayesian dynamic evaluation (intervention date: {dynamic_date})",
        timeout_s=900,
        est_s=20,
        input_text=f"{dynamic_date}\n",
        env_overrides={
            # Keep Bayesian dynamic eval practical in routine pipeline runs.
            "ADSB_BAYES_DYNAMIC_MODE": os.environ.get("ADSB_BAYES_DYNAMIC_MODE", "quick"),
            "ADSB_BAYES_DYNAMIC_DRAWS": str(_env_posint("ADSB_BAYES_DYNAMIC_DRAWS", 120)),
            "ADSB_BAYES_DYNAMIC_TUNE": str(_env_posint("ADSB_BAYES_DYNAMIC_TUNE", 120)),
            "ADSB_BAYES_DYNAMIC_MAX_CHAINS": str(_env_posint("ADSB_BAYES_DYNAMIC_MAX_CHAINS", 1)),
        },
        expected_outputs=[],
    )
    add(
        stage=5,
        script_rel="adsb/analysis/bayesian/adsb_bayesian_advi_eval.py",
        label=f"Bayesian ADVI (intervention date: {dynamic_date})",
        timeout_s=600,
        est_s=60,
        input_text=f"{dynamic_date}\n",
        expected_outputs=[],
    )
    add(
        stage=5,
        script_rel="adsb/analysis/change_points/adsb_detect_multi_change_points.py",
        label="Multiple change-point detection (standard outputs: change_point/multi_change_points_*)",
        timeout_s=300,
        est_s=20,
        env_overrides={
            "JAX_PLATFORMS": "cpu",
            "ADSB_MCP_WARMUP": str(_env_posint("ADSB_MCP_WARMUP", 250)),
            "ADSB_MCP_SAMPLES": str(_env_posint("ADSB_MCP_SAMPLES", 500)),
            "ADSB_MCP_CHAINS": str(_env_posint("ADSB_MCP_CHAINS", 1)),
        },
        depends_on_outputs=["adsb_daily_summary_v2.csv"],
        expected_outputs=[
            "change_point/multi_change_points_report.txt",
            "change_point/multi_change_points_result.json",
        ],
        expected_min_bytes=120,
    )
    add(
        stage=5,
        script_rel="adsb/analysis/change_points/adsb_detect_change_point.py",
        label="Single change-point detection (standard outputs: change_point/change_point_*)",
        timeout_s=300,
        est_s=20,
        env_overrides={
            "JAX_PLATFORMS": "cpu",
            "ADSB_CP_WARMUP": str(_env_posint("ADSB_CP_WARMUP", 200)),
            "ADSB_CP_SAMPLES": str(_env_posint("ADSB_CP_SAMPLES", 400)),
            "ADSB_CP_CHAINS": str(_env_posint("ADSB_CP_CHAINS", 1)),
        },
        depends_on_outputs=["adsb_daily_summary_v2.csv"],
        expected_outputs=[
            "change_point/change_point_report.txt",
            "change_point/change_point_result.json",
        ],
        expected_min_bytes=120,
    )
    add(
        stage=5,
        script_rel="adsb/analysis/gpu/adsb_cuda_evaluator.py",
        label="Change-point visualization (-> performance/adsb_cuda_evaluator_change_point.png)",
        timeout_s=60,
        est_s=10,
        expected_outputs=["performance/adsb_cuda_evaluator_change_point.png"],
        expected_min_bytes=200,
    )
    add(
        stage=5,
        script_rel="adsb/analysis/gpu/adsb_cuda_processor.py",
        label="Change-point visualization #2 (-> performance/adsb_cuda_processor_change_point.png)",
        timeout_s=60,
        est_s=10,
        expected_outputs=["performance/adsb_cuda_processor_change_point.png"],
        expected_min_bytes=200,
    )

    if full_mode:
        add(
            stage=5,
            script_rel="adsb/analysis/change_points/adsb_multi_discovery.py",
            label="Multiple change points (CPU; optional)",
            timeout_s=120,
            est_s=20,
            expected_outputs=[],
        )

    # ================================================================
    # Stage 6: Final Reports
    # ================================================================
    add(
        stage=6,
        script_rel="adsb/analysis/reports/adsb_total_performance_reporter.py",
        label="Consolidated report (-> tsuchiura_master_log_report.png)",
        timeout_s=240,
        est_s=30,
        expected_outputs=["tsuchiura_master_log_report.png"],
        expected_min_bytes=200,
    )

    # ================================================================
    # Stage 7: PLAO (independent data source)
    # ================================================================
    # No data dependency with ADS-B pipeline (Stage 1-6).
    # PLAO pos (schema_ver=1) distance-bin AUC aggregation, stats, and plots.
    if not skip_plao:
        plao_extra_args = ["--plot-mode", "compact"]
        if _env_bool("ARENA_PLAO_PLOTS", True):
            plao_extra_args.insert(0, "--plots")
        else:
            plao_extra_args.insert(0, "--no-plots")
        add(
            stage=7,
            script_rel="plao/analysis/plao_distance_auc_eval.py",
            label="PLAO distance-band AUC evaluation (-> plao/distance_auc/*)",
            timeout_s=900,
            est_s=120,
            extra_args=plao_extra_args,
            skip_if_no_inputs=True,
            input_dir="plao_pos",
            input_pattern="pos_*.jsonl",
            expected_outputs=[
                "plao/distance_auc/plao_daily_distance_auc_summary.csv",
                "plao/distance_auc/plao_daily_distance_auc_long.csv",
                "plao/distance_auc/plao_distance_auc_stats_report.txt",
                "plao/distance_auc/plao_auc_norm_total_trend_compact.png",
                "plao/distance_auc/plao_auc_norm_bins_trend_compact.png",
            ],
            expected_min_bytes=100,
        )

    # ================================================================
    # Stage 8: OpenSky Comparison (independent data source)
    # ================================================================
    # Join OpenSky dist_1m.jsonl with local pos_YYYYMMDD.jsonl and evaluate
    # capture rate, distance-bin coverage, and phase-level improvements.
    # Statistical models: MWU, Bootstrap, NB-GLM, Kruskal-Wallis, distance-bin NB-GLM
    add(
        stage=8,
        script_rel="adsb/analysis/opensky/adsb_opensky_comparison_eval.py",
        label="OpenSky comparison evaluation (-> opensky_comparison/*)",
        timeout_s=1200,
        est_s=180,
        extra_args=["--plots"] if _env_bool("ARENA_OPENSKY_COMPARE_PLOTS", True) else ["--no-plots"],
        expected_outputs=[
            "opensky_comparison/opensky_local_minutely_merged.csv",
            "opensky_comparison/opensky_comparison_daily_summary.csv",
            "opensky_comparison/opensky_comparison_stats_report.txt",
            "opensky_comparison/daily_capture_trend.png",
            "opensky_comparison/daily_bin_capture_trend.png",
            "opensky_comparison/capture_ratio_by_phase.png",
            "opensky_comparison/capture_by_distance_bin.png",
            "opensky_comparison/capture_heatmap_phase_distance.png",
        ],
        expected_min_bytes=100,
    )

    return steps
