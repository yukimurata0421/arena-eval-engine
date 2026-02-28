#!/usr/bin/env python3
"""
pipeline.py - ADS-B Evaluation Framework Pipeline Orchestrator

Purpose:
  - Orchestrates the ADS-B evaluation pipeline (aggregation → spatial → statistics → phase →
    Bayesian/change points → final reports → PLAO) in data-dependency order.
  - Continues on failure, while recording failure reasons and artifact consistency.

Key features:
  - Windows/WSL backend switching (validate outputs via Windows paths even when running in WSL)
  - GPU (JAX) auto-detection (checked inside WSL when running there)
  - Stage selection, single-step runs, dry-run, GPU disable
  - Expected artifact existence/size validation (catches "returncode=0 but broken")
  - Append-only JSONL execution metadata for auditability

Usage (CLI):
  arena run --stage 3
  arena run --only 2
  arena run --dry-run
  arena run --no-gpu
  arena run --full
  arena run --backend auto|native|wsl
  arena run --scripts-root <project>/scripts --output-root <project>/output --data-root <project>/data
  arena validate

Pipeline stages:
  Stage 1: Aggregation      — daily AUC aggregation, traffic merge, signal strength aggregation
  Stage 2: Spatial/Visual   — heatmaps, coverage, LOS efficiency
  Stage 3: Statistics       — NB GLM, distance-bin comparisons, time-of-day, signal quality
  Stage 4: Phase Evaluation — phase Bayesian comparison (NumPyro NUTS)
  Stage 5: Bayesian/CP      — Bayesian dynamic eval, change point detection
  Stage 6: Final Reports    — consolidated report generation
  Stage 7: PLAO             — PLAO pos distance-bin AUC (independent data source)
  Stage 8: OpenSky Compare  — OpenSky vs local reception comparison and tests

Recommendation:
  In public/production usage, keep `pipeline_runs.jsonl` as an evidence log.

Change log:
  v2 → v3 (2026-02-26):
    - Stage 4 speedup: force CPU for all DiscreteHMCGibbs scripts
      (at n=59, GPU was 38-50x slower; 658s/748s → 17s/15s)
    - Removed 3 redundant scripts from full_mode:
      adsb_detect_discovery_gpu.py, adsb_performance_discovery.py, adsb_bayesian_eval.py
    - Stage 4 total: 2488s → ~93s (96% reduction)
    - Whole pipeline: 2717s (45.3min) → ~322s (5.4min)
  v3 → v4 (2026-02-27):
    - Added adsb_phase_evaluator.py as Stage 4 (Phase Eval)
    - Added plao_distance_auc_eval.py as Stage 7 (PLAO)
    - Former Stage 4 (Bayesian/CP) → Stage 5
    - Former Stage 5 (Final Reports) → Stage 6
    - Removed 5 redundant scripts (also from full_mode):
      adsb_bayesian_cuda_eval.py  — wrapper, redundant
      adsb_bayesian_eval.py       — equivalent with CUDA forced to CPU
      adsb_detect_discovery_gpu.py — same model as detect_change_point
      adsb_performance_discovery.py — same model as detect_multi_change_points
      adsb_manual_evaluator.py    — legacy subset of phase_evaluator
    - Added --skip-plao flag
    - Whole pipeline: ~322s → ~340s (+Phase ~20s; PLAO independent)
  v4 → v5 (2026-02-28):
    - Stage 4: adsb_phase_evaluator.py → adsb_phase_evaluator_v3.py (Dual-Baseline)
      Resolves the issue where RTL-SDR → Airspy dominates and masks Cable/Adapter effects.
      Outputs two views: Section 1 (vs Phase 0) + Section 2 (vs Alt Baseline).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from arena.lib.phase_config import load_phase_config
from arena.lib.runtime_config import build_snapshot

# =========================
# Env (batch-friendly)
# =========================
BATCH_ENV = {
    "ADSB_BATCH_MODE": "1",
    "MPLBACKEND": "Agg",  # matplotlib non-interactive backend
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}


# =========================
# Helpers
# =========================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def is_windows() -> bool:
    return os.name == "nt"


def default_roots_native() -> tuple[Path, Path, Path]:
    """
    Paths accessible from *current Python runtime*.
    - Default layout: <project>/scripts, <project>/output, <project>/data
    """
    from arena.lib.paths import ROOT, SCRIPTS_ROOT

    scripts_root = SCRIPTS_ROOT
    project_root = ROOT
    return scripts_root, (project_root / "output"), (project_root / "data")


def _windows_to_wsl_path(p: Path) -> str:
    if os.name != "nt":
        return str(p)
    drive = p.drive[0].lower() if p.drive else "c"
    rest = p.as_posix().split(":", 1)[-1].lstrip("/")
    return f"/mnt/{drive}/{rest}"


def default_roots_exec_for_wsl(
    scripts_root_native: Path,
    output_root_native: Path,
    data_root_native: Path,
) -> tuple[str, str, str]:
    """
    Paths used *inside WSL bash*.
    """
    return (
        _windows_to_wsl_path(scripts_root_native),
        _windows_to_wsl_path(output_root_native),
        _windows_to_wsl_path(data_root_native),
    )


def wsl_available() -> bool:
    if not is_windows():
        return False
    try:
        p = subprocess.run(
            ["wsl", "-e", "bash", "-lc", "command -v python3 >/dev/null 2>&1; echo $?"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return p.returncode == 0 and p.stdout.strip().endswith("0")
    except Exception:
        return False


def tail_text(s: str, max_chars: int = 1200) -> str:
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


# =========================
# Backend abstraction
# =========================
@dataclass
class Backend:
    """
    - native: run scripts with current python (sys.executable) and native file paths.
    - wsl:    run scripts via `wsl bash -lc` using /mnt/<drive>/... paths for execution,
              while validating outputs via Windows native paths (E:\\...) if master runs on Windows.
    """

    kind: str  # "native" or "wsl"
    scripts_root_native: Path
    output_root_native: Path
    data_root_native: Path

    # execution roots used in commands (posix paths inside WSL)
    scripts_root_exec: str | None = None
    output_root_exec: str | None = None
    data_root_exec: str | None = None
    pythonpath_exec: str | None = None

    python_native: str = field(default_factory=lambda: sys.executable)

    def describe(self) -> str:
        if self.kind == "native":
            return f"native ({self.python_native})"
        return "wsl (python3)"

    def ensure_output_dirs(self, subdirs: Sequence[str]) -> None:
        """
        Create output directories on *native filesystem* (Windows path when master runs on Windows).
        """
        for d in subdirs:
            (self.output_root_native / d).mkdir(parents=True, exist_ok=True)

    def run_python_snippet(self, code: str, env: dict[str, str], timeout_s: int = 30) -> subprocess.CompletedProcess:
        """
        Run a python -c snippet in the execution environment (native or wsl).
        """
        if self.kind == "native":
            cmd = [self.python_native, "-c", code]
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)

        # WSL: run python3 -c ...
        code_q = shlex.quote(code)
        cmd = ["wsl", "-e", "bash", "-lc", f"python3 -c {code_q}"]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)

    def build_script_cmd(self, script_rel_posix: str, extra_args: list[str] | None = None) -> tuple[list[str], Optional[str]]:
        """
        Return (cmd, cwd) for subprocess.run.
        - script_rel_posix must be posix-style relative path, e.g. "adsb/aggregators/adsb_aggregator.py"
        - extra_args: additional CLI arguments to pass to the script
        """
        if self.kind == "native":
            script_path = self.scripts_root_native / script_rel_posix
            cmd = [self.python_native, str(script_path)]
            if extra_args:
                cmd.extend(extra_args)
            return cmd, str(self.scripts_root_native)

        assert self.scripts_root_exec is not None
        rel_q = shlex.quote(script_rel_posix)
        args_str = " ".join(shlex.quote(a) for a in extra_args) if extra_args else ""
        env_prefix = ""
        if self.pythonpath_exec:
            env_prefix = f"PYTHONPATH={shlex.quote(self.pythonpath_exec)} "
        payload = f"cd {shlex.quote(self.scripts_root_exec)} && {env_prefix}python3 {rel_q}"
        if args_str:
            payload += f" {args_str}"
        cmd = ["wsl", "-e", "bash", "-lc", payload]
        return cmd, None


# =========================
# Pipeline model
# =========================
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
    expected_min_bytes: int = 50
    extra_args: list[str] = field(default_factory=list)


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
    skip_plao: bool = False


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
    stderr_tail: str = ""
    stdout_tail: str = ""


# =========================
# Validation
# =========================
def _resolve_expected_path(output_root_native: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return output_root_native / p


def validate_outputs(
    output_root_native: Path,
    expected_outputs: Sequence[str],
    min_bytes: int,
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for rel in expected_outputs:
        fp = _resolve_expected_path(output_root_native, rel)
        if not fp.exists():
            missing.append(str(fp))
            continue
        try:
            if fp.is_file() and fp.stat().st_size < min_bytes:
                missing.append(f"{fp} (too small: {fp.stat().st_size} bytes)")
        except Exception:
            missing.append(f"{fp} (stat failed)")
    return (len(missing) == 0), missing


# =========================
# Dependency checks
# =========================
BASE_MODULES = ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "folium", "requests"]
STAGE4_MODULES = ["jax", "numpyro"]         # phase eval (NumPyro NUTS)
STAGE5_MODULES = ["jax", "numpyro"]         # change point / numpyro
STAGE5_PYMC = ["pymc", "arviz"]             # PyMC bayesian scripts


def missing_modules(backend: Backend, modules: Sequence[str], env: dict[str, str]) -> list[str]:
    mods = list(modules)
    code = (
        "import importlib.util as u; "
        f"mods={mods!r}; "
        "miss=[m for m in mods if u.find_spec(m) is None]; "
        "print('\\n'.join(miss))"
    )
    try:
        proc = backend.run_python_snippet(code, env=env, timeout_s=30)
        out = (proc.stdout or "").strip()
        if not out:
            return []
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return list(modules)


# =========================
# GPU detection (JAX)
# =========================
def detect_gpu_jax(backend: Backend, env: dict[str, str]) -> dict:
    """
    Detect GPU availability for JAX in the *execution environment*.
    """
    info = {"available": False, "device": "CPU only", "jax": False}
    code = (
        "import jax; "
        "ds=jax.devices(); "
        "gpu=[d for d in ds if d.platform=='gpu']; "
        "print('1' if len(gpu)>0 else '0'); "
        "print(gpu[0].device_kind if gpu else 'none')"
    )
    try:
        proc = backend.run_python_snippet(code, env={**env, "JAX_PLATFORMS": "cuda,cpu"}, timeout_s=30)
        lines = (proc.stdout or "").strip().splitlines()
        if lines and lines[0].strip() == "1":
            info["available"] = True
            info["jax"] = True
            info["device"] = lines[1].strip() if len(lines) > 1 else "GPU"
    except Exception:
        pass
    return info


# =========================
# Stage name registry
# =========================
STAGE_NAMES = {
    1: "Aggregation",
    2: "Spatial / Visual",
    3: "Statistics (CPU)",
    4: "Phase Evaluation",
    5: "Bayesian & Change Points",
    6: "Final Reports",
    7: "PLAO",
    8: "OpenSky Comparison",
}


# =========================
# Pipeline definition
# =========================
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

    def add(**kwargs) -> None:
        steps.append(Step(**kwargs))

    # ================================================================
    # Stage 1: Aggregation
    # ================================================================
    add(
        stage=1,
        script_rel="adsb/aggregators/adsb_aggregator.py",
        label="Daily AUC aggregation (raw → adsb_daily_summary_raw.csv)",
        timeout_s=300,
        critical=True,
        est_s=60,
        expected_outputs=["adsb_daily_summary_raw.csv"],
        expected_min_bytes=200,
    )
    add(
        stage=1,
        script_rel="adsb/aggregators/adsb_eval_pk_aggregator.py",
        label="Merged daily summary (→ adsb_daily_summary.csv)",
        timeout_s=300,
        critical=True,
        est_s=120,
        expected_outputs=["adsb_daily_summary.csv"],
        expected_min_bytes=500,
    )
    add(
        stage=1,
        script_rel="adsb/aggregators/adsb_csv_patcher.py",
        label="CSV column patching (hardware/is_post_change, etc.)",
        timeout_s=120,
        critical=True,
        est_s=10,
        expected_outputs=["adsb_daily_summary.csv"],
        expected_min_bytes=500,
    )
    add(
        stage=1,
        script_rel="signals/collectors/signal_stats_aggregator.py",
        label="Signal strength aggregation (→ adsb_signal_range_summary.csv)",
        timeout_s=300,
        est_s=60,
        expected_outputs=["adsb_signal_range_summary.csv"],
        expected_min_bytes=20,
    )
    add(
        stage=1,
        script_rel="adsb/data_fetch/get_opensky_traffic.py",
        label="Fetch OpenSky movements (→ data/flight_data/airport_movements.csv)",
        timeout_s=300,
        est_s=30,
        expected_outputs=[str((default_roots_native()[2] / "flight_data" / "airport_movements.csv"))],
        expected_min_bytes=200,
    )
    add(
        stage=1,
        script_rel="adsb/aggregators/adsb_local_traffic_proxy_gen.py",
        label="Generate local_traffic_proxy (→ adsb_daily_summary_v2.csv)",
        timeout_s=900,
        critical=True,
        est_s=300,
        expected_outputs=["adsb_daily_summary_v2.csv"],
        expected_min_bytes=500,
    )
    add(
        stage=1,
        script_rel="adsb/analysis/time_resolved/adsb_time_resolved_aggregator.py",
        label="Time-bin AUC aggregation (→ time_resolved/adsb_timebin_summary.csv)",
        timeout_s=300,
        est_s=120,
        expected_outputs=["time_resolved/adsb_timebin_summary.csv"],
        expected_min_bytes=300,
    )

    # ================================================================
    # Stage 2: Spatial / Visual
    # ================================================================
    add(
        stage=2,
        script_rel="adsb/analysis/reports/adsb_fringe_decoding_evaluator.py",
        label="Long-range decode rate (→ fringe_decoding/*)",
        timeout_s=600,
        est_s=120,
        expected_outputs=[
            "fringe_decoding/fringe_decoding_stats.csv",
            "fringe_decoding/statistical_report.txt",
        ],
        expected_min_bytes=200,
    )
    add(
        stage=2,
        script_rel="adsb/analysis/reports/adsb_polar_coverage_evaluator.py",
        label="Azimuth coverage (→ coverage/*)",
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
        label="LOS efficiency trend (→ vertical_profile/*)",
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
        label="Combined heatmap (→ adsb_coverage_heatmap.html)",
        timeout_s=300,
        est_s=30,
        expected_outputs=["adsb_coverage_heatmap.html"],
        expected_min_bytes=200,
    )
    add(
        stage=2,
        script_rel="adsb/heatmap/adsb_daily_heatmap_alt.py",
        label="Altitude-band heatmaps (→ heatmaps/)",
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
        label="NB GLM (baseline) (→ performance/baseline_nb_*)",
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
        label="Distance-bin comparison (MWU + Bootstrap) (→ performance/distance_performance_summary.csv)",
        timeout_s=240,
        est_s=30,
        expected_outputs=["performance/distance_performance_summary.csv"],
        expected_min_bytes=200,
    )
    add(
        stage=3,
        script_rel="adsb/analysis/stats/adsb_distance_binomial_eval.py",
        label="Distance-bin comparison (Binomial) (→ performance/distance_binomial_summary.csv)",
        timeout_s=240,
        est_s=20,
        expected_outputs=["performance/distance_binomial_summary.csv"],
        expected_min_bytes=200,
    )
    add(
        stage=3,
        script_rel="adsb/analysis/reports/adsb_fringe_decoding_quality_stats.py",
        label="Fringe decode quality (v2; console)",
        timeout_s=180,
        est_s=15,
        expected_outputs=[],
    )
    add(
        stage=3,
        script_rel="adsb/analysis/time_resolved/adsb_time_resolved_mixed_eval.py",
        label="Time-bin mixed model (→ performance/time_resolved_performance.png)",
        timeout_s=180,
        est_s=20,
        expected_outputs=["performance/time_resolved_performance.png"],
        expected_min_bytes=200,
    )
    add(
        stage=3,
        script_rel="adsb/analysis/time_resolved/adsb_time_resolved_detailed_report.py",
        label="Time-bin detailed report (→ performance/time_bin_detailed_stats.csv, etc.)",
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
        script_rel="signals/analysis/signal_range_evaluator.py",
        label="Signal strength by distance bin (console)",
        timeout_s=120,
        est_s=10,
        expected_outputs=[],
    )
    add(
        stage=3,
        script_rel="signals/analysis/signal_quality_evaluator.py",
        label="Signal quality in 150-175km band (console)",
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
        label="Phase Bayesian comparison (dual baseline) (→ performance/phase_evaluator_*)",
        timeout_s=300,
        est_s=30,
        env_overrides={
            "ADSB_BATCH_MODE": "1",
            "ADSB_PHASE_INTERACTIVE": "0",
            "JAX_PLATFORMS": "cpu",
        },
        expected_outputs=[
            "performance/phase_evaluator_results.csv",
            "performance/phase_evaluator_report.txt",
            "performance/phase_evaluator_boxplot.png",
        ],
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
        label="Bayesian phase comparison (→ performance/bayesian_phase_results_cuda.csv)",
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
        est_s=90,
        input_text=f"{dynamic_date}\n",
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
        label="Multi change point detection (K=3; console/plots)",
        timeout_s=120,
        est_s=20,
        expected_outputs=[],
    )
    add(
        stage=5,
        script_rel="adsb/analysis/change_points/adsb_detect_change_point.py",
        label="Single change point detection (console/plots)",
        timeout_s=120,
        est_s=20,
        expected_outputs=[],
    )
    add(
        stage=5,
        script_rel="adsb/analysis/gpu/adsb_cuda_evaluator.py",
        label="Change point visualization (→ performance/adsb_cuda_evaluator_change_point.png)",
        timeout_s=60,
        est_s=10,
        expected_outputs=["performance/adsb_cuda_evaluator_change_point.png"],
        expected_min_bytes=200,
    )
    add(
        stage=5,
        script_rel="adsb/analysis/gpu/adsb_cuda_processor.py",
        label="Change point visualization #2 (→ performance/adsb_cuda_processor_change_point.png)",
        timeout_s=60,
        est_s=10,
        expected_outputs=["performance/adsb_cuda_processor_change_point.png"],
        expected_min_bytes=200,
    )

    if full_mode:
        add(
            stage=5,
            script_rel="adsb/analysis/change_points/adsb_multi_discovery.py",
            label="Multi change points (CPU; optional)",
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
        label="Consolidated report (→ tsuchiura_master_log_report.png)",
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
        add(
            stage=7,
            script_rel="plao/analysis/plao_distance_auc_eval.py",
            label="PLAO distance-bin AUC evaluation (→ plao/distance_auc/*)",
            timeout_s=900,
            est_s=120,
            extra_args=["--plots", "--plot-mode", "compact"],
            expected_outputs=[
                "plao/distance_auc/plao_daily_distance_auc_summary.csv",
                "plao/distance_auc/plao_daily_distance_auc_long.csv",
                "plao/distance_auc/plao_distance_auc_stats_report.txt",
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
        label="OpenSky comparison evaluation (→ opensky_comparison/*)",
        timeout_s=1200,
        est_s=180,
        extra_args=["--plots"],
        expected_outputs=[
            "opensky_comparison/opensky_local_minutely_merged.csv",
            "opensky_comparison/opensky_comparison_daily_summary.csv",
            "opensky_comparison/opensky_comparison_stats_report.txt",
        ],
        expected_min_bytes=100,
    )

    return steps


# =========================
# Runner
# =========================
class PipelineRunner:
    def __init__(
        self,
        backend: Backend,
        dry_run: bool,
        validate: bool,
        jsonl_log_path: Path,
        jax_platforms: str,
        skip_existing: bool,
        fail_fast: bool,
        phase_config_path: str = "",
    ) -> None:
        self._phase_config_path = phase_config_path
        self.backend = backend
        self.dry_run = dry_run
        self.validate = validate
        self.jsonl_log_path = jsonl_log_path
        self.skip_existing = skip_existing
        self.fail_fast = fail_fast

        src_root = self.backend.scripts_root_native.parent / "src"
        if src_root.exists():
            existing = os.environ.get("PYTHONPATH", "")
            py_path = f"{src_root}{os.pathsep}{existing}" if existing else str(src_root)
        else:
            py_path = os.environ.get("PYTHONPATH", "")
        self.env = {
            **os.environ,
            **BATCH_ENV,
            "JAX_PLATFORMS": jax_platforms,
            "ADSB_PHASE_CONFIG": self._phase_config_path,
            "PYTHONPATH": py_path,
        }
        self.records: list[RunRecord] = []

        # ensure log dir exists
        self.jsonl_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, rec: RunRecord) -> None:
        payload = {
            "ts_start": rec.ts_start,
            "ts_end": rec.ts_end,
            "backend": rec.backend,
            "stage": rec.stage,
            "label": rec.label,
            "script": rec.script_rel,
            "status": rec.status,
            "elapsed_s": rec.elapsed_s,
            "returncode": rec.returncode,
            "cmd": rec.cmd,
            "expected_outputs": rec.expected_outputs,
            "outputs_ok": rec.outputs_ok,
            "missing_outputs": rec.missing_outputs,
            "stderr_tail": rec.stderr_tail,
            "stdout_tail": rec.stdout_tail,
        }
        with self.jsonl_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_config_snapshot(self, snapshot: dict) -> None:
        payload = {
            "ts": now_iso(),
            "status": "CONFIG",
            "backend": self.backend.describe(),
            "config_snapshot": snapshot,
        }
        with self.jsonl_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _should_skip_existing(self, step: Step) -> bool:
        if not self.skip_existing or not step.expected_outputs:
            return False
        ok, _missing = validate_outputs(self.backend.output_root_native, step.expected_outputs, step.expected_min_bytes)
        return ok

    def run_step(self, step: Step) -> bool:
        # normalize script path to posix
        script_rel = step.script_rel.replace("\\", "/")

        # skip existing outputs (optional)
        if self._should_skip_existing(step):
            rec = RunRecord(
                ts_start=now_iso(),
                ts_end=now_iso(),
                backend=self.backend.describe(),
                stage=step.stage,
                label=step.label,
                script_rel=script_rel,
                status="SKIP(existing)",
                elapsed_s=0.0,
                returncode=None,
                cmd=[],
                expected_outputs=list(step.expected_outputs),
                outputs_ok=True,
                missing_outputs=[],
            )
            self.records.append(rec)
            self._append_jsonl(rec)
            print(f"    - [SKIP] {step.label} (outputs already exist)")
            return True

        cmd, cwd = self.backend.build_script_cmd(script_rel, step.extra_args or None)

        # dry-run
        if self.dry_run:
            rec = RunRecord(
                ts_start=now_iso(),
                ts_end=now_iso(),
                backend=self.backend.describe(),
                stage=step.stage,
                label=step.label,
                script_rel=script_rel,
                status="DRY",
                elapsed_s=0.0,
                returncode=None,
                cmd=cmd,
                expected_outputs=list(step.expected_outputs),
                outputs_ok=True,
                missing_outputs=[],
            )
            self.records.append(rec)
            self._append_jsonl(rec)
            est = f" (~{step.est_s}s)" if step.est_s else ""
            print(f"    [DRY] {step.label}{est}: {script_rel}")
            return True

        # skip PLAO step if no input files (independent data source)
        if script_rel.endswith("plao/analysis/plao_distance_auc_eval.py"):
            input_dir = self.backend.data_root_native / "plao_pos"
            pattern = "pos_*.jsonl"
            if step.extra_args:
                for i, a in enumerate(step.extra_args):
                    if a == "--input-dir" and i + 1 < len(step.extra_args):
                        input_dir = Path(step.extra_args[i + 1])
                    elif a == "--pattern" and i + 1 < len(step.extra_args):
                        pattern = step.extra_args[i + 1]

            has_inputs = False
            try:
                has_inputs = any(input_dir.glob(pattern))
            except Exception:
                has_inputs = False

            if not has_inputs:
                rec = RunRecord(
                    ts_start=now_iso(),
                    ts_end=now_iso(),
                    backend=self.backend.describe(),
                    stage=step.stage,
                    label=step.label,
                    script_rel=script_rel,
                    status="SKIP(no input)",
                    elapsed_s=0.0,
                    returncode=None,
                    cmd=[],
                    expected_outputs=list(step.expected_outputs),
                    outputs_ok=True,
                    missing_outputs=[],
                )
                self.records.append(rec)
                self._append_jsonl(rec)
                print(f"    - [SKIP] {step.label} (no input files in {input_dir})")
                return True

        # existence check for native backend
        if self.backend.kind == "native":
            sp = self.backend.scripts_root_native / script_rel
            if not sp.exists():
                rec = RunRecord(
                    ts_start=now_iso(),
                    ts_end=now_iso(),
                    backend=self.backend.describe(),
                    stage=step.stage,
                    label=step.label,
                    script_rel=script_rel,
                    status="NOT_FOUND",
                    elapsed_s=0.0,
                    returncode=None,
                    cmd=cmd,
                    expected_outputs=list(step.expected_outputs),
                    outputs_ok=False,
                    missing_outputs=[str(sp)],
                    stderr_tail="",
                    stdout_tail="",
                )
                self.records.append(rec)
                self._append_jsonl(rec)
                print(f"    ? [NOT_FOUND] {step.label}: {script_rel}")
                return (not step.critical) and (not self.fail_fast)

        # run
        t0 = time.time()
        ts0 = now_iso()
        est = f" (~{step.est_s}s)" if step.est_s else ""
        print(f"    > {step.label}{est} ...", end=" ", flush=True)

        try:
            env = {**self.env, **(step.env_overrides or {})}
            if self.backend.kind == "wsl" and step.env_overrides:
                exports = "; ".join(
                    [f"export {k}={shlex.quote(v)}" for k, v in step.env_overrides.items()]
                )
                cmd = list(cmd)
                cmd[-1] = f"{exports}; {cmd[-1]}"
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                input=step.input_text,
                capture_output=True,
                text=True,
                timeout=step.timeout_s,
                encoding="utf-8",
                errors="replace",
            )
            elapsed = time.time() - t0
            ts1 = now_iso()

            status = "OK" if proc.returncode == 0 else "FAIL"
            outputs_ok = True
            missing = []

            if self.validate and step.expected_outputs:
                outputs_ok, missing = validate_outputs(
                    self.backend.output_root_native,
                    step.expected_outputs,
                    step.expected_min_bytes,
                )
                if status == "OK" and not outputs_ok:
                    status = "FAIL_OUTPUT"

            rec = RunRecord(
                ts_start=ts0,
                ts_end=ts1,
                backend=self.backend.describe(),
                stage=step.stage,
                label=step.label,
                script_rel=script_rel,
                status=status,
                elapsed_s=round(elapsed, 2),
                returncode=proc.returncode,
                cmd=cmd,
                expected_outputs=list(step.expected_outputs),
                outputs_ok=outputs_ok,
                missing_outputs=missing,
                stderr_tail=tail_text(proc.stderr or "", 1400),
                stdout_tail=tail_text(proc.stdout or "", 1400),
            )
            self.records.append(rec)
            self._append_jsonl(rec)

            if status == "OK":
                print(f"OK（{elapsed:.1f}s）")
                return True

            # failure printing (tail)
            print(f"NG（{elapsed:.1f}s）")
            if status == "FAIL_OUTPUT" and missing:
                print("      [output validation failed]")
                for m in missing[:3]:
                    print(f"      - {m}")
                if len(missing) > 3:
                    print(f"      ... (+{len(missing)-3} more)")
            if proc.stderr:
                lines = proc.stderr.strip().splitlines()
                for line in lines[-3:]:
                    print(f"      {line}")

            # control flow
            if step.critical or self.fail_fast:
                return False
            return True

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            ts1 = now_iso()
            rec = RunRecord(
                ts_start=ts0,
                ts_end=ts1,
                backend=self.backend.describe(),
                stage=step.stage,
                label=step.label,
                script_rel=script_rel,
                status="TIMEOUT",
                elapsed_s=round(elapsed, 2),
                returncode=None,
                cmd=cmd,
                expected_outputs=list(step.expected_outputs),
                outputs_ok=False,
                missing_outputs=[],
                stderr_tail="",
                stdout_tail="",
            )
            self.records.append(rec)
            self._append_jsonl(rec)
            print(f"NG（{step.timeout_s}s）")
            return (not step.critical) and (not self.fail_fast)

        except Exception as e:
            elapsed = time.time() - t0
            ts1 = now_iso()
            rec = RunRecord(
                ts_start=ts0,
                ts_end=ts1,
                backend=self.backend.describe(),
                stage=step.stage,
                label=step.label,
                script_rel=script_rel,
                status="ERROR",
                elapsed_s=round(elapsed, 2),
                returncode=None,
                cmd=cmd,
                expected_outputs=list(step.expected_outputs),
                outputs_ok=False,
                missing_outputs=[],
                stderr_tail=str(e),
                stdout_tail="",
            )
            self.records.append(rec)
            self._append_jsonl(rec)
            print(f"ERROR ({e})")
            return (not step.critical) and (not self.fail_fast)

    def print_summary(self) -> None:
        print("\n" + "=" * 78)
        print("Pipeline Summary")
        print("=" * 78)

        total = sum(r.elapsed_s for r in self.records)
        counts: dict[str, int] = {}
        for r in self.records:
            k = r.status.split("(")[0]  # SKIP(x) -> SKIP
            counts[k] = counts.get(k, 0) + 1

        for r in self.records:
            icon = {
                "OK": "OK",
                "FAIL": "NG",
                "FAIL_OUTPUT": "NG*",
                "TIMEOUT": "TO",
                "ERROR": "!!",
                "NOT_FOUND": "?",
                "DRY": "DRY",
            }.get(r.status.split("(")[0], "-")
            t = f"{r.elapsed_s:7.1f}s" if r.elapsed_s > 0 else "        "
            stage_name = STAGE_NAMES.get(r.stage, f"Stage {r.stage}")
            print(f"  {icon} {t}  [{r.status:<12}] S{r.stage}({stage_name}) {r.label}")

        print("-" * 78)
        parts = []
        for key in ["OK", "FAIL", "FAIL_OUTPUT", "TIMEOUT", "ERROR", "SKIP", "DRY", "NOT_FOUND"]:
            if key in counts:
                parts.append(f"{counts[key]} {key}")
        print("  " + " / ".join(parts))
        print(f"  Total time: {total:.0f}s ({total/60:.1f} min)")
        print(f"  Log: {self.jsonl_log_path}")
        print("=" * 78)


# =========================
# Runner
# =========================
def run(cfg: RunConfig) -> int:
    # roots (native)
    sr_def, or_def, dr_def = default_roots_native()
    scripts_root_native = Path(cfg.scripts_root) if cfg.scripts_root else sr_def

    # --- Phase config resolution ---
    if cfg.phase_config:
        phase_config_path = str(Path(cfg.phase_config).resolve())
    else:
        phase_config_path = str(scripts_root_native / "config" / "phases.txt")

    # dynamic-date: if empty, load from config
    dynamic_date = cfg.dynamic_date
    if not dynamic_date:
        try:
            _pcfg = load_phase_config(phase_config_path)
            dynamic_date = _pcfg.intervention_date
        except Exception:
            dynamic_date = "2026-02-11"
    output_root_native = Path(cfg.output_root) if cfg.output_root else or_def
    data_root_native = Path(cfg.data_root) if cfg.data_root else dr_def

    # backend choice
    backend_kind = "native"
    if cfg.backend == "native":
        backend_kind = "native"
    elif cfg.backend == "wsl":
        backend_kind = "wsl"
    else:
        if is_windows() and wsl_available():
            backend_kind = "wsl"
        else:
            backend_kind = "native"

    if backend_kind == "wsl" and (not is_windows()) and os.name != "posix":
        backend_kind = "native"

    # Build backend with proper mapping
    if backend_kind == "native":
        backend = Backend(
            kind="native",
            scripts_root_native=scripts_root_native,
            output_root_native=output_root_native,
            data_root_native=data_root_native,
        )
    else:
        s_exec, o_exec, d_exec = default_roots_exec_for_wsl(
            scripts_root_native,
            output_root_native,
            data_root_native,
        )
        py_exec = _windows_to_wsl_path(scripts_root_native.parent / "src")
        backend = Backend(
            kind="wsl",
            scripts_root_native=scripts_root_native,
            output_root_native=output_root_native,
            data_root_native=data_root_native,
            scripts_root_exec=s_exec,
            output_root_exec=o_exec,
            data_root_exec=d_exec,
            pythonpath_exec=py_exec,
        )

    # header
    print("=" * 78)
    print("ADS-B Evaluation Framework - pipeline")
    print(f"Time:    {now_iso()}")
    print(f"Backend: {backend.describe()}")
    print(f"Native scripts: {backend.scripts_root_native}")
    print(f"Native output:  {backend.output_root_native}")
    print(f"Native data:    {backend.data_root_native}")
    if backend.kind == "wsl":
        print(f"WSL scripts:    {backend.scripts_root_exec}")
        print(f"WSL output:     {backend.output_root_exec}")
        print(f"WSL data:       {backend.data_root_exec}")
    print(f"Phase config:   {phase_config_path}")
    print(f"Dynamic date:   {dynamic_date}")
    print(f"Skip PLAO:      {cfg.skip_plao}")
    print("=" * 78)

    # build pipeline
    steps = build_pipeline(
        dynamic_date=dynamic_date,
        full_mode=cfg.full,
        skip_plao=cfg.skip_plao,
    )

    # deps check based on planned stages
    planned_stages: set[int] = set()
    for s in steps:
        planned_stages.add(s.stage)

    # env
    env = {**os.environ, **BATCH_ENV}

    # required modules
    req = list(BASE_MODULES)
    if not cfg.dry_run:
        if 4 in planned_stages:
            req += STAGE4_MODULES
        if 5 in planned_stages and not cfg.no_gpu:
            req += STAGE5_MODULES
            req += STAGE5_PYMC
    miss = missing_modules(backend, req, env=env)
    if miss:
        print("\n[ERROR] Missing python modules in execution environment:")
        for m in miss:
            print(f"  - {m}")
        print("\nFix:")
        if backend.kind == "native":
            print("  pip install -e \".[dev]\"  (or install modules listed above)")
        else:
            print("  In WSL: pip3 install -e \".[dev]\"  (or install modules listed above)")
        return 1

    # GPU detect (in execution environment)
    if cfg.no_gpu:
        jax_platforms = "cpu"
        print("\nGPU: disabled (--no-gpu)")
    elif cfg.dry_run:
        jax_platforms = "cuda,cpu"
        print("\nGPU: (dry-run) skip detection")
    else:
        print("\nGPU: detecting ...", end=" ", flush=True)
        gpu_info = detect_gpu_jax(backend, env=env)
        if gpu_info["available"]:
            print(f"OK ({gpu_info['device']})")
            jax_platforms = "cuda,cpu"
        else:
            print("not found -> CPU")
            jax_platforms = "cpu"

    # output dirs
    output_root_native.mkdir(parents=True, exist_ok=True)
    backend.ensure_output_dirs(
        [
            "performance",
            "coverage",
            "fringe_decoding",
            "heatmaps",
            "vertical_profile",
            "time_resolved",
            "plao/distance_auc",
            "opensky_comparison",
        ]
    )

    # log path
    if cfg.log_jsonl:
        log_jsonl = Path(cfg.log_jsonl)
    else:
        log_jsonl = output_root_native / "performance" / "pipeline_runs.jsonl"

    runner = PipelineRunner(
        backend=backend,
        dry_run=cfg.dry_run,
        validate=cfg.validate,
        jsonl_log_path=log_jsonl,
        jax_platforms=jax_platforms,
        skip_existing=cfg.skip_existing,
        fail_fast=cfg.fail_fast,
        phase_config_path=phase_config_path,
    )
    runner.log_config_snapshot(build_snapshot(phase_config_path))

    # validate-only mode
    if cfg.validate_only:
        print("\n[validate-only] Checking expected artifacts in pipeline definition ...")
        all_ok = True
        for st in sorted(set(s.stage for s in steps)):
            stage_name = STAGE_NAMES.get(st, f"Stage {st}")
            print(f"\n  Stage {st} ({stage_name})")
            for step in [x for x in steps if x.stage == st]:
                if not step.expected_outputs:
                    print(f"    - {step.label}: (no expected outputs)")
                    continue
                ok, missing = validate_outputs(output_root_native, step.expected_outputs, step.expected_min_bytes)
                if ok:
                    print(f"    OK  {step.label}")
                else:
                    all_ok = False
                    print(f"    NG  {step.label}")
                    for m in missing[:3]:
                        print(f"       - {m}")
                    if len(missing) > 3:
                        print(f"       ... (+{len(missing)-3} more)")
        return 0 if all_ok else 1

    # execution plan filter
    def should_run_stage(n: int) -> bool:
        if cfg.only is not None:
            return n == cfg.only
        return n >= cfg.stage

    start = time.time()
    ok_all = True

    for st in sorted(set(s.stage for s in steps)):
        if not should_run_stage(st):
            continue
        stage_name = STAGE_NAMES.get(st, f"Stage {st}")
        print("\n" + "-" * 78, flush=True)
        print(f"Stage {st}: {stage_name}", flush=True)
        print("-" * 78, flush=True)
        for step in [x for x in steps if x.stage == st]:
            ok = runner.run_step(step)
            if not ok:
                ok_all = False
                if cfg.fail_fast or step.critical:
                    print("\n[STOP] Critical failure or --fail-fast.")
                    runner.print_summary()
                    return 1

    runner.print_summary()
    elapsed = time.time() - start
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # exit code
    if not ok_all:
        return 1

    # treat any NG as non-zero
    ng = [r for r in runner.records if r.status in ("FAIL", "FAIL_OUTPUT", "TIMEOUT", "ERROR", "NOT_FOUND")]
    if ng:
        print(f"\nWARN: {len(ng)} failures exist. See log: {log_jsonl}")
        return 1

    return 0
