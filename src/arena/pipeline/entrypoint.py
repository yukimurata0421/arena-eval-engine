from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from arena.lib.config_resolution import build_runtime_config_metadata, validate_resolved_config_paths
from arena.lib.phase_config import load_phase_config
from arena.lib.runtime_config import build_snapshot
from arena.log import get_logger
from arena.pipeline.backend import (
    BASE_MODULES,
    BATCH_ENV,
    STAGE4_MODULES,
    STAGE5_MODULES,
    STAGE5_PYMC,
    Backend,
    _windows_to_wsl_path,
    default_roots_exec_for_wsl,
    default_roots_native,
    detect_gpu_jax,
    is_windows,
    missing_modules,
    now_iso,
    resolve_default_workers,
    wsl_available,
    wsl_path_is_dir,
)
from arena.pipeline.runner import PipelineRunner
from arena.pipeline.stages import (
    STAGE_NAMES,
    RunConfig,
    Step,
    build_pipeline,
    resolve_pipeline_build_options,
    validate_outputs,
)

logger = get_logger(__name__)


CHANGE_POINT_REQUIRED_OUTPUTS = [
    "change_point/change_point_report.txt",
    "change_point/multi_change_points_report.txt",
]

PARALLEL_STAGES = {2, 3, 5}
# Wave dependency chain: 0→1→3→6, and 0→7.  Steps 2/4/5 are independent.
# OpenSky (5) moved to Wave 4 so its ~150s API call doesn't block Wave 3.
STAGE1_WAVES: list[list[int]] = [[0], [1, 2, 4], [3], [5, 6, 7]]
# Stage 3 stats scripts consume Stage 2 outputs (for example fringe_decoding stats),
# so keep Stage 3 outside this cross-stage parallel group to avoid read-before-produce races.
PARALLEL_STAGE_GROUPS = [(2, 4, 5, 7, 8)]
# Stages that use independent data sources and can launch during Stage 1
EARLY_LAUNCH_STAGES = {7}  # PLAO: reads plao_pos/, no Stage 1 dependency

# Used only when dynamic_date is not provided and phase config cannot be loaded.
# Keep as a named constant so the intent is visible in diffs/reviews.
DEFAULT_INTERVENTION_DATE_FOR_DEV = "2026-02-11"


# ------------------------------------------------------------------
# Parallel / sequential execution helpers
# ------------------------------------------------------------------
def _collect_futures(
    futures: dict[Future, Step],
    fail_fast: bool,
    runner: PipelineRunner,
) -> bool:
    """Collect results from parallel step futures. Returns False if a critical failure occurred."""
    ok_all = True
    for future in as_completed(futures):
        step = futures[future]
        try:
            ok = future.result()
            if not ok:
                ok_all = False
                if fail_fast or step.critical:
                    logger.error("\n[Stopped] Stopped due to critical failure or --fail-fast.")
                    runner.print_summary()
                    return False
        except Exception as exc:
            ok_all = False
            logger.error("\n    [ERROR] %s: %s", step.label, exc)
            if fail_fast or step.critical:
                runner.print_summary()
                return False
    return ok_all


def _run_steps_parallel(
    runner: PipelineRunner,
    steps: list[Step],
    max_workers: int,
    fail_fast: bool,
) -> tuple[bool, bool]:
    """Run steps in parallel. Returns (ok_all, should_abort)."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(runner.run_step, step): step for step in steps}
        ok = _collect_futures(futures, fail_fast, runner)
    if not ok:
        # Distinguish "some failures but no abort" from "abort requested"
        any_critical = any(s.critical for s in steps)
        should_abort = fail_fast or any_critical
        return False, should_abort
    return True, False


def _run_steps_sequential(
    runner: PipelineRunner,
    steps: list[Step],
    fail_fast: bool,
) -> tuple[bool, bool]:
    """Run steps sequentially. Returns (ok_all, should_abort)."""
    ok_all = True
    for step in steps:
        ok = runner.run_step(step)
        if not ok:
            ok_all = False
            if fail_fast or step.critical:
                logger.error("\n[Stopped] Stopped due to critical failure or --fail-fast.")
                runner.print_summary()
                return False, True
    return ok_all, False


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def run(cfg: RunConfig) -> int:
    sr_def, or_def, dr_def = default_roots_native()
    scripts_root_native = Path(cfg.scripts_root) if cfg.scripts_root else sr_def

    config_meta = build_runtime_config_metadata(
        settings_override=cfg.settings_path or None,
        phase_override=cfg.phase_config or None,
        scripts_root=scripts_root_native,
        analysis_start_date=os.getenv("ARENA_ANALYSIS_START_DATE", ""),
        analysis_end_date=os.getenv("ARENA_ANALYSIS_END_DATE", ""),
        env={},
    )
    config_errors = validate_resolved_config_paths(config_meta)
    if config_errors:
        logger.error("[Error] config resolution failed:")
        for err in config_errors:
            logger.error("  - %s", err)
        return 1

    phase_config_path = str(config_meta["resolved_phase_config_path"])
    settings_path = str(config_meta["resolved_settings_path"])

    dynamic_date = cfg.dynamic_date
    if not dynamic_date:
        try:
            _pcfg = load_phase_config(phase_config_path)
            dynamic_date = _pcfg.intervention_date
        except (OSError, ValueError, KeyError) as exc:
            dynamic_date = DEFAULT_INTERVENTION_DATE_FOR_DEV
            logger.warning(
                "[WARN] Failed to load phases.txt (%s): %s: %s" " - using fallback date=%s",
                phase_config_path,
                type(exc).__name__,
                exc,
                dynamic_date,
            )
    output_root_native = Path(cfg.output_root) if cfg.output_root else or_def
    data_root_native = Path(cfg.data_root) if cfg.data_root else dr_def

    backend = _build_backend(cfg, scripts_root_native, output_root_native, data_root_native)

    _print_header(backend, settings_path, phase_config_path, config_meta, dynamic_date, cfg)
    resolved_workers = cfg.workers if cfg.workers > 0 else resolve_default_workers()

    steps = build_pipeline(
        dynamic_date=dynamic_date,
        full_mode=cfg.full,
        skip_plao=cfg.skip_plao,
        options=resolve_pipeline_build_options(os.environ),
    )

    def should_run_stage(n: int) -> bool:
        if cfg.only is not None:
            return n == cfg.only
        return n >= cfg.stage

    planned_stages: set[int] = set()
    for s in steps:
        if should_run_stage(s.stage):
            planned_stages.add(s.stage)

    env = {
        **os.environ,
        **BATCH_ENV,
        "ARENA_PHASE_CONFIG": phase_config_path,
        "ADSB_PHASE_CONFIG": phase_config_path,
        "ARENA_SETTINGS": settings_path,
        "ADSB_SETTINGS": settings_path,
    }

    req = list(BASE_MODULES)
    if not cfg.dry_run:
        if 4 in planned_stages:
            req += STAGE4_MODULES
        if 5 in planned_stages and not cfg.no_gpu:
            req += STAGE5_MODULES
            req += STAGE5_PYMC
    miss = missing_modules(backend, req, env=env)
    if miss:
        logger.error("\n[ERROR] Missing Python module in execution environment:")
        for m in miss:
            logger.error("  - %s", m)
        logger.error("\nAction:")
        if backend.kind == "native":
            logger.error(' pip install -e ".[dev]" (or install the above modules individually)')
        else:
            logger.error(' Within WSL: pip3 install -e ".[dev]" (or install the above modules separately)')
        return 1

    jax_platforms = _detect_gpu(cfg, backend, env)

    output_root_native.mkdir(parents=True, exist_ok=True)
    backend.ensure_output_dirs(
        [
            "performance",
            "change_point",
            "coverage",
            "fringe_decoding",
            "heatmaps",
            "vertical_profile",
            "time_resolved",
            "plao/distance_auc",
            "opensky_comparison",
        ]
    )

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
        settings_path=settings_path,
        workers=cfg.workers,
        steps=steps,
    )
    runner.log_config_snapshot(
        build_snapshot(
            phase_config_path,
            settings_path=settings_path,
            scripts_root=str(scripts_root_native),
            resolution_metadata=config_meta,
        )
    )

    if cfg.validate_only:
        return _run_validate_only(steps, output_root_native, data_root_native)

    start = time.time()
    ok_all = _run_pipeline_stages(
        runner,
        steps,
        cfg,
        should_run_stage,
        resolved_workers,
    )

    runner.print_summary()
    err_report = runner.write_error_code_report()
    logger.info("Error code report: %s", err_report)
    elapsed = time.time() - start
    logger.info("\nTotal elapsed time: %.0fs (%.1f min)", elapsed, elapsed / 60)

    if 5 in planned_stages and not _check_change_point_contract(output_root_native, data_root_native):
        return 1

    if not ok_all:
        return 1

    ng = [r for r in runner.records if r.status in ("FAIL", "FAIL_OUTPUT", "TIMEOUT", "ERROR", "NOT_FOUND")]
    if ng:
        logger.warning("\nWarning: %d failures. Log: %s", len(ng), log_jsonl)
        return 1

    return 0


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _build_backend(
    cfg: RunConfig,
    scripts_root_native: Path,
    output_root_native: Path,
    data_root_native: Path,
) -> Backend:
    backend_kind = "native"
    if cfg.backend == "native":
        backend_kind = "native"
    elif cfg.backend == "wsl":
        backend_kind = "wsl"
    else:
        if is_windows() and wsl_available():
            wsl_ready, missing = _wsl_required_paths_accessible(scripts_root_native, output_root_native, data_root_native)
            if not wsl_ready:
                logger.warning("[WARN] WSL is available, but project paths are not accessible inside WSL; falling back to native backend.")
                for item in missing[:4]:
                    logger.warning("  - %s", item)
            else:
                backend_kind = "wsl"

    if cfg.backend == "wsl" and is_windows():
        wsl_ready, missing = _wsl_required_paths_accessible(scripts_root_native, output_root_native, data_root_native)
        if not wsl_ready:
            logger.warning("[WARN] WSL backend requested, but some project paths are not accessible inside WSL.")
            for item in missing[:4]:
                logger.warning("  - %s", item)

    if backend_kind == "wsl" and (not is_windows()) and os.name != "posix":
        backend_kind = "native"

    if backend_kind == "native":
        return Backend(
            kind="native",
            scripts_root_native=scripts_root_native,
            output_root_native=output_root_native,
            data_root_native=data_root_native,
        )

    s_exec, o_exec, d_exec = default_roots_exec_for_wsl(
        scripts_root_native,
        output_root_native,
        data_root_native,
    )
    py_exec = _windows_to_wsl_path(scripts_root_native.parent / "src")
    return Backend(
        kind="wsl",
        scripts_root_native=scripts_root_native,
        output_root_native=output_root_native,
        data_root_native=data_root_native,
        scripts_root_exec=s_exec,
        output_root_exec=o_exec,
        data_root_exec=d_exec,
        pythonpath_exec=py_exec,
    )


def _wsl_required_paths_accessible(
    scripts_root_native: Path,
    output_root_native: Path,
    data_root_native: Path,
) -> tuple[bool, list[str]]:
    checks = [
        ("scripts root", scripts_root_native),
        ("data root", data_root_native),
    ]
    output_probe = output_root_native if output_root_native.exists() else output_root_native.parent
    checks.append(("output root" if output_root_native.exists() else "output parent", output_probe))

    missing: list[str] = []
    for label, path in checks:
        if not wsl_path_is_dir(path):
            missing.append(f"{label}: {_windows_to_wsl_path(path)}")
    return not missing, missing


def _print_header(
    backend: Backend,
    settings_path: str,
    phase_config_path: str,
    config_meta: dict,
    dynamic_date: str,
    cfg: RunConfig,
) -> None:
    resolved_workers = cfg.workers if cfg.workers > 0 else resolve_default_workers()
    logger.info("=" * 78)
    logger.info("ADS-B Evaluation Framework - Pipeline")
    logger.info("Time: %s", now_iso())
    logger.info("Backend: %s", backend.describe())
    logger.info("Native scripts: %s", backend.scripts_root_native)
    logger.info("Native output: %s", backend.output_root_native)
    logger.info("Native data: %s", backend.data_root_native)
    logger.info("settings:      %s", settings_path)
    if backend.kind == "wsl":
        logger.info("WSL scripts:    %s", backend.scripts_root_exec)
        logger.info("WSL output:     %s", backend.output_root_exec)
        logger.info("WSL data:       %s", backend.data_root_exec)
    logger.info("Phase configuration: %s", phase_config_path)
    logger.info(
        "config default: settings=%d phase=%d experimental=%d",
        int(bool(config_meta["used_default_settings"])),
        int(bool(config_meta["used_default_phase_config"])),
        int(bool(config_meta["experimental_mode"])),
    )
    if str(config_meta.get("analysis_start_date", "")):
        logger.info("analysis_start_date: %s", config_meta.get("analysis_start_date"))
    if str(config_meta.get("analysis_end_date", "")):
        logger.info("analysis_end_date: %s", config_meta.get("analysis_end_date"))
    logger.info("Dynamic date: %s", dynamic_date)
    logger.info("PLAO Skip: %s", cfg.skip_plao)
    logger.info("Parallel workers: %s", resolved_workers)
    logger.info("=" * 78)


def _detect_gpu(cfg: RunConfig, backend: Backend, env: dict[str, str]) -> str:
    if cfg.no_gpu:
        logger.info("\nGPU: Disable (--no-gpu)")
        return "cpu"
    if cfg.dry_run:
        logger.info("\nGPU: Skip (dry-run) detection")
        return "cuda,cpu"
    logger.info("\nGPU: Detecting...")
    gpu_info = detect_gpu_jax(backend, env=env)
    if gpu_info["available"]:
        logger.info("GPU: OK (%s)", gpu_info["device"])
        return "cuda,cpu"
    if gpu_info.get("reason"):
        logger.info("GPU: Not available to JAX -> CPU (%s: %s)", gpu_info.get("device", "CPU only"), gpu_info["reason"])
    else:
        logger.info("GPU: Not detected -> CPU")
    return "cpu"


def _run_validate_only(
    steps: list[Step],
    output_root_native: Path,
    data_root_native: Path,
) -> int:
    logger.info("\n[validate-only] Checking expected artifacts of pipeline definition...")
    all_ok = True
    for st in sorted(set(s.stage for s in steps)):
        stage_name = STAGE_NAMES.get(st, f"Stage {st}")
        logger.info("\n  Stage %d (%s)", st, stage_name)
        for step in [x for x in steps if x.stage == st]:
            if not step.expected_outputs:
                logger.info(" - %s: (no expected output)", step.label)
                continue
            ok, missing = validate_outputs(
                output_root_native,
                step.expected_outputs,
                step.expected_min_bytes,
                data_root_native=data_root_native,
            )
            if ok:
                logger.info("    OK  %s", step.label)
            else:
                all_ok = False
                logger.error("    NG  %s", step.label)
                for m in missing[:3]:
                    logger.error("       - %s", m)
                if len(missing) > 3:
                    logger.error("       ... (+%d more)", len(missing) - 3)
    return 0 if all_ok else 1


def _run_pipeline_stages(
    runner: PipelineRunner,
    steps: list[Step],
    cfg: RunConfig,
    should_run_stage,
    resolved_workers: int,
) -> bool:
    ok_all = True
    stages_already_run_in_group: set[int] = set()

    early_futures, early_executor, early_stages = _launch_early_stages(
        runner,
        steps,
        should_run_stage,
        resolved_workers,
    )

    for st in sorted(set(s.stage for s in steps)):
        if not should_run_stage(st):
            continue
        if st in stages_already_run_in_group or st in early_stages:
            continue

        group_stages = next((g for g in PARALLEL_STAGE_GROUPS if g[0] == st), None)
        if group_stages is not None and resolved_workers > 1:
            group_ok = _run_stage_group(
                runner,
                steps,
                group_stages,
                should_run_stage,
                resolved_workers,
                cfg.fail_fast,
                st,
                exclude_stages=early_stages,
            )
            if group_ok is None:
                continue
            if not group_ok:
                _collect_early(early_futures, early_executor, cfg.fail_fast, runner)
                return False
            stages_already_run_in_group.update(group_stages)
            continue

        stage_ok = _run_single_stage(
            runner,
            steps,
            st,
            resolved_workers,
            cfg.fail_fast,
        )
        if not stage_ok:
            ok_all = False
            if cfg.fail_fast or any(step.critical for step in steps if step.stage == st):
                _collect_early(early_futures, early_executor, cfg.fail_fast, runner)
                return False

    if not _collect_early(early_futures, early_executor, cfg.fail_fast, runner):
        ok_all = False

    return ok_all


def _run_stage_group(
    runner: PipelineRunner,
    steps: list[Step],
    group_stages: tuple[int, ...],
    should_run_stage,
    resolved_workers: int,
    fail_fast: bool,
    current_st: int,
    exclude_stages: set[int] | None = None,
) -> bool | None:
    """Run a group of stages in parallel. Returns None to skip, True/False for success/failure."""
    exclude = exclude_stages or set()
    group_stages_to_run = [s for s in group_stages if should_run_stage(s) and s not in exclude]
    if len(group_stages_to_run) == 1 and group_stages_to_run[0] != current_st:
        return None
    if len(group_stages_to_run) < 2:
        return None

    group_steps = [x for x in steps if x.stage in group_stages and should_run_stage(x.stage) and x.stage not in exclude]
    names = " / ".join(STAGE_NAMES.get(s, f"Stage {s}") for s in group_stages if s not in exclude)
    group_est_min = sum(s.est_s for s in group_steps) / 60.0
    max_workers = min(resolved_workers, len(group_steps))

    logger.info("\n" + "-" * 78)
    logger.info("Stage %s: %s", ", ".join(str(s) for s in group_stages if s not in exclude), names)
    logger.info("-" * 78)
    logger.info(" (estimated approximately %.0f minutes | parallel %d steps, workers=%d)", group_est_min, len(group_steps), max_workers)

    ok, should_abort = _run_steps_parallel(runner, group_steps, max_workers, fail_fast)
    if should_abort:
        return False
    return ok


def _run_single_stage(
    runner: PipelineRunner,
    steps: list[Step],
    st: int,
    resolved_workers: int,
    fail_fast: bool,
) -> bool:
    """Run a single stage. Returns True if all steps succeeded."""
    stage_name = STAGE_NAMES.get(st, f"Stage {st}")
    logger.info("\n" + "-" * 78)
    logger.info("Stage %d: %s", st, stage_name)
    logger.info("-" * 78)
    stage_steps = [x for x in steps if x.stage == st]

    # Stage 1: wave-based parallelism
    if st == 1 and STAGE1_WAVES and stage_steps:
        _validate_stage1_wave_indices(stage_steps)
        if len(stage_steps) >= max(max(w) for w in STAGE1_WAVES) + 1 and resolved_workers > 1:
            return _run_stage1_waves(runner, stage_steps, resolved_workers, fail_fast)

    use_parallel = st in PARALLEL_STAGES and len(stage_steps) > 1 and resolved_workers > 1
    stage_est_min = sum(s.est_s for s in stage_steps) / 60.0

    if use_parallel:
        max_workers = min(resolved_workers, len(stage_steps))
        logger.info(" (estimated approximately %.0f minutes | parallel %d steps, workers=%d)", stage_est_min, len(stage_steps), max_workers)
        ok, should_abort = _run_steps_parallel(runner, stage_steps, max_workers, fail_fast)
        if should_abort:
            return False
        return ok
    else:
        logger.info(" (estimated approximately %.0f minutes | sequential %d steps)", stage_est_min, len(stage_steps))
        ok, should_abort = _run_steps_sequential(runner, stage_steps, fail_fast)
        if should_abort:
            return False
        return ok


def _validate_stage1_wave_indices(stage_steps: list[Step]) -> None:
    # Prefer stable "wave" metadata if available, fall back to historical index-based checks.
    if any(getattr(s, "wave", 0) for s in stage_steps):
        expected = {
            "adsb_aggregator": 1,
            "adsb_csv_patcher": 3,
            "get_opensky_traffic": 4,
            "local_traffic_proxy": 4,
        }
        for kw, w in expected.items():
            hit = next((s for s in stage_steps if kw in s.script_rel), None)
            if hit is None:
                logger.warning("Stage 1 wave: Step not found for expected keyword '%s'.", kw)
                continue
            if (hit.wave or 0) != w:
                logger.warning(
                    "Stage 1 wave: '%s' is not wave=%d (actually wave=%s, script=%s).",
                    kw,
                    w,
                    hit.wave,
                    hit.script_rel,
                )
        return

    wave_expected = {0: "adsb_aggregator", 3: "adsb_csv_patcher", 5: "get_opensky_traffic", 6: "local_traffic_proxy"}
    for wi, kw in wave_expected.items():
        if wi < len(stage_steps) and kw not in stage_steps[wi].script_rel:
            logger.warning(
                "STAGE1_WAVES index %d does not match expected keyword '%s'"
                " (Actual: %s). Please reflect the reordering of build_pipeline() in STAGE1_WAVES.",
                wi,
                kw,
                stage_steps[wi].script_rel,
            )


def _launch_early_stages(
    runner: PipelineRunner,
    steps: list[Step],
    should_run_stage,
    resolved_workers: int,
) -> tuple[dict[Future, Step], ThreadPoolExecutor | None, set[int]]:
    """Launch EARLY_LAUNCH_STAGES concurrently with Stage 1.

    Returns (futures_dict, executor, set_of_launched_stage_numbers).
    Only activates when Stage 1 is also planned (otherwise no overlap benefit).
    """
    if resolved_workers <= 1 or not should_run_stage(1):
        return {}, None, set()
    early_steps = [s for s in steps if s.stage in EARLY_LAUNCH_STAGES and should_run_stage(s.stage)]
    if not early_steps:
        return {}, None, set()
    executor = ThreadPoolExecutor(max_workers=min(resolved_workers, len(early_steps)))
    futures: dict[Future, Step] = {executor.submit(runner.run_step, step): step for step in early_steps}
    launched = {s.stage for s in early_steps}
    for st in sorted(launched):
        n = sum(1 for s in early_steps if s.stage == st)
        logger.info(
            "\n(Pre-parallel startup) Stage %d: %s (%d step — concurrent execution with Stage 1)",
            st,
            STAGE_NAMES.get(st, f"Stage {st}"),
            n,
        )
    return futures, executor, launched


def _collect_early(
    futures: dict[Future, Step],
    executor: ThreadPoolExecutor | None,
    fail_fast: bool,
    runner: PipelineRunner,
) -> bool:
    """Collect results from early-launched stages."""
    if not futures:
        return True
    ok = _collect_futures(futures, fail_fast, runner)
    if executor is not None:
        executor.shutdown(wait=False)
    return ok


def _run_stage1_waves(
    runner: PipelineRunner,
    stage_steps: list[Step],
    resolved_workers: int,
    fail_fast: bool,
) -> bool:
    stage_est_min = sum(s.est_s for s in stage_steps) / 60.0
    logger.info(" (estimated approximately %.0f minutes | wave parallel)", stage_est_min)
    ok_all = True

    # Prefer stable "wave" metadata if available, otherwise fall back to STAGE1_WAVES indices.
    if any(getattr(s, "wave", 0) for s in stage_steps):
        wave_keys = sorted({int(s.wave) for s in stage_steps if int(getattr(s, "wave", 0) or 0) > 0})
        wave_plan: list[tuple[int, list[Step]]] = [(w, [s for s in stage_steps if int(s.wave or 0) == w]) for w in wave_keys]
    else:
        wave_plan = [(wave_idx, [stage_steps[i] for i in wave_indices if i < len(stage_steps)]) for wave_idx, wave_indices in enumerate(STAGE1_WAVES, 1)]

    for wave_idx, wave_steps in wave_plan:
        if not wave_steps:
            continue
        wave_est = sum(s.est_s for s in wave_steps)
        if len(wave_steps) > 1:
            logger.info(" --- Wave %d: %d step parallel (estimated approximately %ds) ---", wave_idx, len(wave_steps), wave_est)
            ok, should_abort = _run_steps_parallel(
                runner,
                wave_steps,
                min(resolved_workers, len(wave_steps)),
                fail_fast,
            )
            if should_abort:
                return False
            if not ok:
                ok_all = False
        else:
            logger.info(" --- Wave %d: 1 step (prediction approx. %ds) ---", wave_idx, wave_est)
            ok, should_abort = _run_steps_sequential(runner, wave_steps, fail_fast)
            if should_abort:
                return False
            if not ok:
                ok_all = False
    return ok_all


def _check_change_point_contract(output_root_native: Path, data_root_native: Path) -> bool:
    cp_ok, cp_missing = validate_outputs(
        output_root_native,
        CHANGE_POINT_REQUIRED_OUTPUTS,
        min_bytes=80,
        data_root_native=data_root_native,
    )
    cp_contract_path = output_root_native / "performance" / "change_point_contract_latest.txt"
    cp_lines = [
        f"generated_at: {now_iso()}",
        "pipeline_stage5_planned: 1",
        f"change_point_required_ok: {int(cp_ok)}",
        "required_outputs:",
        *[f"- {x}" for x in CHANGE_POINT_REQUIRED_OUTPUTS],
        "missing_outputs:",
    ]
    if cp_missing:
        cp_lines.extend([f"- {x}" for x in cp_missing])
    else:
        cp_lines.append("- (none)")
    cp_contract_path.parent.mkdir(parents=True, exist_ok=True)
    cp_contract_path.write_text("\n".join(cp_lines) + "\n", encoding="utf-8")
    logger.info("change-point deliverable contract check: %s", cp_contract_path)
    if not cp_ok:
        logger.error("[ERROR] change-point required artifact is missing.")
        for m in cp_missing:
            logger.error("  - %s", m)
        return False
    return True
