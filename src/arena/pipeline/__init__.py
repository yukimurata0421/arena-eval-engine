"""ARENA pipeline package.

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

from arena.pipeline.entrypoint import run
from arena.pipeline.stages import RunConfig, RunRecord, Step

__all__ = ["run", "RunConfig", "Step", "RunRecord"]
