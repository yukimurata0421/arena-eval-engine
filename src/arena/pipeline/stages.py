from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Step:
    stage: int
    script_rel: str
    label: str
    # Stage 1 uses wave-based parallelism; wave=0 means "unspecified".
    wave: int = 0
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
    settings_path: str = ""
    validate: bool = True
    validate_only: bool = False
    skip_existing: bool = False
    fail_fast: bool = False
    log_jsonl: str = ""
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
                        missing.append(f"{fp} (stale: latest mtime {latest:.0f} < {min_mtime:.0f})")
                continue

            if fp.is_file() and fp.stat().st_size < min_bytes:
                missing.append(f"{fp} (too small: {fp.stat().st_size} bytes)")
                continue
            if min_mtime is not None and fp.is_file() and fp.stat().st_mtime < min_mtime:
                missing.append(f"{fp} (stale: mtime {fp.stat().st_mtime:.0f} < {min_mtime:.0f})")
        except OSError:
            missing.append(f"{fp} (stat failed)")
    return (len(missing) == 0), missing


STAGE_NAMES = {
    1: "集計",
    2: "空間/可視化",
    3: "統計 (CPU)",
    4: "フェーズ評価",
    5: "ベイズ/変化点",
    6: "最終レポート",
    7: "PLAO",
    8: "OpenSky 比較",
}


@dataclass(frozen=True)
class PipelineBuildOptions:
    """Resolved build-time options for pipeline step definitions.

    This intentionally separates environment-variable resolution from the pipeline
    definition itself, improving testability and making step definitions deterministic
    under explicit inputs.
    """

    phase_warmup: int = 1000
    phase_samples: int = 2000

    bayes_dynamic_mode: str = "quick"
    bayes_dynamic_draws: int = 120
    bayes_dynamic_tune: int = 120
    # 0 means: use ARENA_MAX_WORKERS / orchestrator-provided parallelism.
    bayes_dynamic_max_chains: int = 0

    plao_plots: bool = True
    opensky_compare_plots: bool = True


def _env_bool(env: dict[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_posint(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return val if val > 0 else default


def resolve_pipeline_build_options(env: dict[str, str] | None = None) -> PipelineBuildOptions:
    e = env if env is not None else dict(os.environ)
    return PipelineBuildOptions(
        phase_warmup=_env_posint(e, "ADSB_PHASE_WARMUP", 1000),
        phase_samples=_env_posint(e, "ADSB_PHASE_SAMPLES", 2000),
        bayes_dynamic_mode=str(e.get("ADSB_BAYES_DYNAMIC_MODE", "quick") or "quick"),
        bayes_dynamic_draws=_env_posint(e, "ADSB_BAYES_DYNAMIC_DRAWS", 120),
        bayes_dynamic_tune=_env_posint(e, "ADSB_BAYES_DYNAMIC_TUNE", 120),
        bayes_dynamic_max_chains=_env_posint(e, "ADSB_BAYES_DYNAMIC_MAX_CHAINS", 0),
        plao_plots=_env_bool(e, "ARENA_PLAO_PLOTS", True),
        opensky_compare_plots=_env_bool(e, "ARENA_OPENSKY_COMPARE_PLOTS", True),
    )


def build_stage1_steps() -> list[Step]:
    return [
        Step(
            stage=1,
            wave=1,
            script_rel="adsb/aggregators/adsb_aggregator.py",
            label="日次AUC集計（raw → adsb_daily_summary_raw.csv）",
            timeout_s=300,
            critical=True,
            est_s=60,
            always_run_when_skip_existing=True,
            expected_outputs=["adsb_daily_summary_raw.csv"],
            expected_min_bytes=200,
        ),
        Step(
            stage=1,
            wave=2,
            script_rel="adsb/aggregators/adsb_eval_pk_aggregator.py",
            label="日次サマリ結合（→ adsb_daily_summary.csv）",
            timeout_s=300,
            critical=True,
            est_s=120,
            always_run_when_skip_existing=True,
            expected_outputs=["adsb_daily_summary.csv"],
            expected_min_bytes=500,
        ),
        Step(
            stage=1,
            wave=2,
            script_rel="adsb/ops/dist_1m_health_check.py",
            label="運用監視: dist_1m 収集/ローテーション健全性チェック（→ performance/dist_1m_health_*）",
            timeout_s=180,
            # Monitoring should not block aggregation/report refresh.
            critical=False,
            est_s=20,
            always_run_when_skip_existing=True,
            soft_fail_on_error=True,
            # Archive rotation is intentionally inactive in this environment.
            # Disable rotation age warnings/failures for pipeline runs.
            extra_args=["--rotation-warn-hours", "0", "--rotation-fail-hours", "0"],
            expected_outputs=[
                "performance/dist_1m_health_latest.json",
                "performance/dist_1m_health_daily.csv",
            ],
            expected_min_bytes=100,
        ),
        Step(
            stage=1,
            wave=3,
            script_rel="adsb/aggregators/adsb_csv_patcher.py",
            label="CSV列パッチ（hardware/is_post_change など）",
            timeout_s=120,
            critical=True,
            est_s=10,
            always_run_when_skip_existing=True,
            expected_outputs=["adsb_daily_summary.csv"],
            expected_min_bytes=500,
        ),
        Step(
            stage=1,
            wave=2,
            script_rel="signals/collectors/signal_stats_aggregator.py",
            label="信号強度集計（→ adsb_signal_range_summary.csv）",
            timeout_s=300,
            est_s=60,
            always_run_when_skip_existing=True,
            expected_outputs=["adsb_signal_range_summary.csv"],
            expected_min_bytes=20,
        ),
        Step(
            stage=1,
            wave=4,
            script_rel="adsb/data_fetch/get_opensky_traffic.py",
            label="OpenSky 移動データ取得（→ data/flight_data/airport_movements.csv）",
            timeout_s=300,
            est_s=30,
            always_run_when_skip_existing=True,
            soft_fail_on_error=True,
            soft_fail_accept_stale_outputs=True,
            expected_outputs=["data://flight_data/airport_movements.csv"],
            expected_min_bytes=200,
        ),
        Step(
            stage=1,
            wave=4,
            script_rel="adsb/aggregators/adsb_local_traffic_proxy_gen.py",
            label="local_traffic_proxy 生成（→ adsb_daily_summary_v2.csv）",
            timeout_s=900,
            critical=True,
            est_s=300,
            always_run_when_skip_existing=True,
            expected_outputs=["adsb_daily_summary_v2.csv"],
            expected_min_bytes=500,
        ),
        Step(
            stage=1,
            wave=4,
            script_rel="adsb/analysis/time_resolved/adsb_time_resolved_aggregator.py",
            label="時間帯AUC集計（→ time_resolved/adsb_timebin_summary.csv）",
            timeout_s=300,
            est_s=120,
            always_run_when_skip_existing=True,
            expected_outputs=["time_resolved/adsb_timebin_summary.csv"],
            expected_min_bytes=300,
        ),
    ]


def build_stage2_steps(*, full_mode: bool) -> list[Step]:
    steps = [
        Step(
            stage=2,
            script_rel="adsb/analysis/reports/adsb_fringe_decoding_evaluator.py",
            label="遠距離デコード率（→ fringe_decoding/*）",
            timeout_s=600,
            est_s=120,
            expected_outputs=[
                "fringe_decoding/fringe_decoding_stats.csv",
            ],
            expected_min_bytes=200,
        ),
        Step(
            stage=2,
            script_rel="adsb/analysis/reports/adsb_polar_coverage_evaluator.py",
            label="方位カバレッジ（→ coverage/*）",
            timeout_s=600,
            est_s=180,
            expected_outputs=[
                "coverage/coverage_trend.csv",
                "coverage/coverage_trend_report.png",
            ],
            expected_min_bytes=200,
        ),
        Step(
            stage=2,
            script_rel="adsb/analysis/reports/adsb_vertical_profile_evaluator.py",
            label="LOS効率トレンド（→ vertical_profile/*）",
            timeout_s=600,
            est_s=180,
            expected_outputs=[
                "vertical_profile/los_efficiency_trend.csv",
                "vertical_profile/los_efficiency_trend_report.png",
            ],
            expected_min_bytes=200,
        ),
        Step(
            stage=2,
            script_rel="adsb/heatmap/adsb_heatmap_generator.py",
            label="統合ヒートマップ（→ adsb_coverage_heatmap.html）",
            timeout_s=300,
            est_s=30,
            expected_outputs=["adsb_coverage_heatmap.html"],
            expected_min_bytes=200,
        ),
        Step(
            stage=2,
            script_rel="adsb/heatmap/adsb_daily_heatmap_alt.py",
            label="高度帯ヒートマップ（→ heatmaps/）",
            timeout_s=900,
            est_s=300,
            expected_outputs=[],
        ),
    ]
    if full_mode:
        steps.append(
            Step(
                stage=2,
                script_rel="adsb/heatmap/adsb_daily_heatmap.py",
                label="日次ヒートマップ（簡易）",
                timeout_s=900,
                est_s=300,
                expected_outputs=[],
            )
        )
    return steps


def build_stage3_steps(*, dynamic_date: str) -> list[Step]:
    return [
        Step(
            stage=3,
            script_rel="adsb/analysis/stats/adsb_baseline_nb_eval.py",
            label="NB GLM（ベースライン）（→ performance/baseline_nb_*）",
            timeout_s=180,
            est_s=20,
            expected_outputs=[
                "performance/baseline_nb_summary.txt",
                "performance/baseline_nb_results.json",
            ],
            expected_min_bytes=150,
        ),
        Step(
            stage=3,
            script_rel="adsb/analysis/stats/adsb_stats_eval.py",
            label="統計評価（statsmodels; コンソール）",
            timeout_s=180,
            est_s=20,
            expected_outputs=[],
        ),
        Step(
            stage=3,
            script_rel="adsb/analysis/stats/adsb_dynamic_eval.py",
            label=f"動的評価（介入日: {dynamic_date}）",
            timeout_s=180,
            est_s=20,
            input_text=f"{dynamic_date}\n",
            expected_outputs=[],
        ),
        Step(
            stage=3,
            script_rel="adsb/analysis/stats/adsb_distance_nb_eval.py",
            label="距離帯比較（MWU + ブートストラップ）（→ performance/distance_performance_summary.csv）",
            timeout_s=240,
            est_s=30,
            expected_outputs=["performance/distance_performance_summary.csv"],
            expected_min_bytes=200,
        ),
        Step(
            stage=3,
            script_rel="adsb/analysis/stats/adsb_distance_binomial_eval.py",
            label="距離帯比較（2項）（→ performance/distance_binomial_summary.csv）",
            timeout_s=240,
            est_s=20,
            expected_outputs=["performance/distance_binomial_summary.csv"],
            expected_min_bytes=200,
        ),
        Step(
            stage=3,
            script_rel="adsb/analysis/reports/adsb_fringe_decoding_quality_stats.py",
            label="フリンジデコード品質（v2; コンソール）",
            timeout_s=180,
            est_s=15,
            expected_outputs=[],
        ),
        Step(
            stage=3,
            script_rel="adsb/analysis/time_resolved/adsb_time_resolved_mixed_eval.py",
            label="時間帯混合モデル（→ performance/time_resolved_performance.png）",
            timeout_s=180,
            est_s=20,
            expected_outputs=["performance/time_resolved_performance.png"],
            expected_min_bytes=200,
        ),
        Step(
            stage=3,
            script_rel="adsb/analysis/time_resolved/adsb_time_resolved_detailed_report.py",
            label="時間帯詳細レポート（→ performance/time_bin_detailed_stats.csv など）",
            timeout_s=180,
            est_s=20,
            expected_outputs=[
                "performance/time_bin_detailed_stats.csv",
                "performance/time_resolved_detailed_plot.png",
            ],
            expected_min_bytes=200,
        ),
        Step(
            stage=3,
            script_rel="adsb/analysis/phase/adsb_phase_timebin_export.py",
            label="日別phase/configマッピング + phase別time-bin集約（→ performance/phase_*）",
            timeout_s=180,
            est_s=20,
            expected_outputs=[
                "performance/phase_config_daily_mapping.csv",
                "performance/phase_timebin_summary.csv",
            ],
            expected_min_bytes=120,
        ),
        Step(
            stage=3,
            script_rel="signals/analysis/signal_range_evaluator.py",
            label="距離帯別信号強度（コンソール）",
            timeout_s=120,
            est_s=10,
            expected_outputs=[],
        ),
        Step(
            stage=3,
            script_rel="signals/analysis/signal_quality_evaluator.py",
            label="150-175km帯の信号品質（コンソール）",
            timeout_s=120,
            est_s=10,
            expected_outputs=[],
        ),
    ]


def build_stage4_steps(*, options: PipelineBuildOptions) -> list[Step]:
    return [
        Step(
            stage=4,
            script_rel="adsb/analysis/phase/adsb_phase_evaluator_v3.py",
            label="フェーズ・ベイズ比較（デュアル基準）（→ performance/phase_evaluator_*）",
            timeout_s=300,
            est_s=30,
            env_overrides={
                "ADSB_BATCH_MODE": "1",
                "ADSB_PHASE_INTERACTIVE": "0",
                "JAX_PLATFORMS": "cpu",
                # Default keeps production-quality sampling, but allow explicit env override for fast runs.
                "ADSB_PHASE_WARMUP": str(options.phase_warmup),
                "ADSB_PHASE_SAMPLES": str(options.phase_samples),
            },
            expected_outputs=[
                "performance/phase_evaluator_results.csv",
                "performance/phase_evaluator_report.txt",
                "performance/phase_evaluator_boxplot.png",
            ],
            depends_on_outputs=["adsb_daily_summary.csv"],
            expected_min_bytes=100,
        )
    ]


def build_stage5_steps(*, dynamic_date: str, full_mode: bool, options: PipelineBuildOptions) -> list[Step]:
    steps = [
        Step(
            stage=5,
            script_rel="adsb/analysis/gpu/adsb_bayesian_phase_cuda_eval.py",
            label="ベイズ・フェーズ比較（→ performance/bayesian_phase_results_cuda.csv）",
            timeout_s=300,
            est_s=20,
            env_overrides={"JAX_PLATFORMS": "cpu"},
            expected_outputs=["performance/bayesian_phase_results_cuda.csv"],
            expected_min_bytes=200,
        ),
        Step(
            stage=5,
            script_rel="adsb/analysis/bayesian/adsb_bayesian_dynamic_eval.py",
            label=f"ベイズ動的評価（介入日: {dynamic_date}）",
            timeout_s=900,
            est_s=20,
            input_text=f"{dynamic_date}\n",
            env_overrides={
                # Keep Bayesian dynamic eval practical in routine pipeline runs.
                "ADSB_BAYES_DYNAMIC_MODE": options.bayes_dynamic_mode,
                "ADSB_BAYES_DYNAMIC_DRAWS": str(options.bayes_dynamic_draws),
                "ADSB_BAYES_DYNAMIC_TUNE": str(options.bayes_dynamic_tune),
                # MAX_CHAINS: 0 or unset = use ARENA_MAX_WORKERS (--workers); 1+ = explicit.
                "ADSB_BAYES_DYNAMIC_MAX_CHAINS": str(options.bayes_dynamic_max_chains),
            },
            expected_outputs=[],
        ),
        Step(
            stage=5,
            script_rel="adsb/analysis/bayesian/adsb_bayesian_advi_eval.py",
            label=f"ベイズ ADVI（介入日: {dynamic_date}）",
            timeout_s=600,
            est_s=60,
            input_text=f"{dynamic_date}\n",
            expected_outputs=[],
        ),
        Step(
            stage=5,
            script_rel="adsb/analysis/change_points/adsb_detect_multi_change_points.py",
            label="複数変化点検出（標準成果物: change_point/multi_change_points_*）",
            timeout_s=300,
            est_s=20,
            env_overrides={"JAX_PLATFORMS": "cpu"},
            depends_on_outputs=["adsb_daily_summary_v2.csv"],
            expected_outputs=[
                "change_point/multi_change_points_report.txt",
                "change_point/multi_change_points_result.json",
            ],
            expected_min_bytes=120,
        ),
        Step(
            stage=5,
            script_rel="adsb/analysis/change_points/adsb_detect_change_point.py",
            label="単一変化点検出（標準成果物: change_point/change_point_*）",
            timeout_s=300,
            est_s=20,
            env_overrides={"JAX_PLATFORMS": "cpu"},
            depends_on_outputs=["adsb_daily_summary_v2.csv"],
            expected_outputs=[
                "change_point/change_point_report.txt",
                "change_point/change_point_result.json",
            ],
            expected_min_bytes=120,
        ),
        Step(
            stage=5,
            script_rel="adsb/analysis/change_points/adsb_daily_signature_change_points.py",
            label="日次 signature change point（→ performance/change_points/daily_signature_cp_*）",
            timeout_s=1200,
            est_s=180,
            depends_on_outputs=["adsb_daily_summary_v2.csv"],
            expected_outputs=["performance/change_points"],
            expected_min_bytes=1,
        ),
        Step(
            stage=5,
            script_rel="adsb/analysis/gpu/adsb_cuda_evaluator.py",
            label="変化点可視化（→ performance/adsb_cuda_evaluator_change_point.png）",
            timeout_s=300,
            est_s=30,
            expected_outputs=["performance/adsb_cuda_evaluator_change_point.png"],
            expected_min_bytes=200,
        ),
        Step(
            stage=5,
            script_rel="adsb/analysis/gpu/adsb_cuda_processor.py",
            label="変化点可視化 #2（→ performance/adsb_cuda_processor_change_point.png）",
            timeout_s=300,
            est_s=30,
            expected_outputs=["performance/adsb_cuda_processor_change_point.png"],
            expected_min_bytes=200,
        ),
    ]
    if full_mode:
        steps.append(
            Step(
                stage=5,
                script_rel="adsb/analysis/change_points/adsb_multi_discovery.py",
                label="複数変化点（CPU; 任意）",
                timeout_s=120,
                est_s=20,
                expected_outputs=[],
            )
        )
    return steps


def build_stage6_steps() -> list[Step]:
    return [
        Step(
            stage=6,
            script_rel="adsb/analysis/reports/adsb_total_performance_reporter.py",
            label="統合レポート（→ tsuchiura_master_log_report.png）",
            timeout_s=240,
            est_s=30,
            expected_outputs=["tsuchiura_master_log_report.png"],
            expected_min_bytes=200,
        )
    ]


def build_stage7_steps(*, options: PipelineBuildOptions) -> list[Step]:
    plao_extra_args = ["--plot-mode", "compact"]
    if options.plao_plots:
        plao_extra_args.insert(0, "--plots")
    else:
        plao_extra_args.insert(0, "--no-plots")
    return [
        Step(
            stage=7,
            script_rel="plao/analysis/plao_distance_auc_eval.py",
            label="PLAO 距離帯AUC評価（→ plao/distance_auc/*）",
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
    ]


def build_stage8_steps(*, options: PipelineBuildOptions) -> list[Step]:
    return [
        Step(
            stage=8,
            script_rel="adsb/analysis/opensky/adsb_opensky_comparison_eval.py",
            label="OpenSky 比較評価（→ opensky_comparison/*）",
            timeout_s=1200,
            est_s=180,
            extra_args=["--plots"] if options.opensky_compare_plots else ["--no-plots"],
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
    ]


def build_pipeline(
    dynamic_date: str,
    full_mode: bool,
    skip_plao: bool = False,
    *,
    options: PipelineBuildOptions | None = None,
) -> list[Step]:
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
    opts = options or resolve_pipeline_build_options()
    steps: list[Step] = []
    steps.extend(build_stage1_steps())
    steps.extend(build_stage2_steps(full_mode=full_mode))
    steps.extend(build_stage3_steps(dynamic_date=dynamic_date))
    steps.extend(build_stage4_steps(options=opts))
    steps.extend(build_stage5_steps(dynamic_date=dynamic_date, full_mode=full_mode, options=opts))
    steps.extend(build_stage6_steps())
    if not skip_plao:
        steps.extend(build_stage7_steps(options=opts))
    steps.extend(build_stage8_steps(options=opts))

    stage_seq: dict[int, int] = {}
    for st in steps:
        stage = int(st.stage)
        stage_seq[stage] = stage_seq.get(stage, 0) + 1
        if not st.error_code_base:
            st.error_code_base = f"S{stage}-{stage_seq[stage]:02d}"

    return steps
