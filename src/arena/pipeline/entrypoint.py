from __future__ import annotations

import os
import time
from pathlib import Path

from arena.lib.phase_config import load_phase_config
from arena.lib.runtime_config import build_snapshot
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
)
from arena.pipeline.runner import PipelineRunner
from arena.pipeline.stages import STAGE_NAMES, RunConfig, build_pipeline, validate_outputs

CHANGE_POINT_REQUIRED_OUTPUTS = [
    "change_point/change_point_report.txt",
    "change_point/multi_change_points_report.txt",
]


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
    print("ADS-B Evaluation Framework - Pipeline")
    print(f"Timestamp:      {now_iso()}")
    print(f"Backend:        {backend.describe()}")
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
    resolved_workers = cfg.workers if cfg.workers > 0 else resolve_default_workers()
    print(f"Workers:        {resolved_workers}")
    print("=" * 78)

    # build pipeline
    steps = build_pipeline(
        dynamic_date=dynamic_date,
        full_mode=cfg.full,
        skip_plao=cfg.skip_plao,
    )

    def should_run_stage(n: int) -> bool:
        if cfg.only is not None:
            return n == cfg.only
        return n >= cfg.stage

    # deps check based on planned stages
    planned_stages: set[int] = set()
    for s in steps:
        if should_run_stage(s.stage):
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
        uniq_miss = list(dict.fromkeys(miss))
        missing_set = set(uniq_miss)
        needs_stage4 = bool(missing_set.intersection(STAGE4_MODULES))
        needs_stage5_pymc = bool(missing_set.intersection(STAGE5_PYMC))

        if needs_stage5_pymc:
            install_hint_native = 'pip install -e ".[dev,bayes,gpu]"'
            install_hint_wsl = 'pip3 install -e ".[dev,bayes,gpu]"'
        elif needs_stage4:
            install_hint_native = 'pip install -e ".[dev,gpu]"'
            install_hint_wsl = 'pip3 install -e ".[dev,gpu]"'
        else:
            install_hint_native = 'pip install -e ".[dev]"'
            install_hint_wsl = 'pip3 install -e ".[dev]"'

        print("\n[ERROR] Missing Python modules in the execution environment:")
        for m in uniq_miss:
            print(f"  - {m}")
        print("\nSuggested action:")
        if backend.kind == "native":
            print(f"  {install_hint_native}  (or install the missing modules individually)")
        else:
            print(f"  In WSL: {install_hint_wsl}  (or install the missing modules individually)")
        return 1

    # GPU detect (in execution environment)
    if cfg.no_gpu:
        jax_platforms = "cpu"
        print("\nGPU: disabled (--no-gpu)")
    elif cfg.dry_run:
        jax_platforms = "cuda,cpu"
        print("\nGPU: skipping detection in dry-run mode")
    else:
        print("\nGPU: probing ...", end=" ", flush=True)
        try:
            gpu_info = detect_gpu_jax(backend, env=env)
        except Exception as exc:
            print(f"probe failed -> CPU ({exc})")
            jax_platforms = "cpu"
        else:
            if gpu_info["available"]:
                print(f"OK ({gpu_info['device']})")
                jax_platforms = "cuda,cpu"
            else:
                print("not detected -> CPU")
                jax_platforms = "cpu"

    # output dirs
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

    # log path
    if cfg.log_jsonl:
        log_jsonl = Path(cfg.log_jsonl)
    else:
        log_jsonl = output_root_native / "performance" / "pipeline_runs.jsonl"
    log_mode = (cfg.log_jsonl_mode or "append").strip().lower()
    if log_mode not in {"append", "overwrite"}:
        print(f"[ERROR] invalid --log-jsonl-mode: {cfg.log_jsonl_mode} (use append|overwrite)")
        return 1
    if log_mode == "overwrite":
        log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        log_jsonl.write_text("", encoding="utf-8")
        print(f"Log mode: overwrite ({log_jsonl})")
    else:
        print(f"Log mode: append ({log_jsonl})")

    runner = PipelineRunner(
        backend=backend,
        dry_run=cfg.dry_run,
        validate=cfg.validate,
        jsonl_log_path=log_jsonl,
        jax_platforms=jax_platforms,
        skip_existing=cfg.skip_existing,
        fail_fast=cfg.fail_fast,
        phase_config_path=phase_config_path,
        workers=cfg.workers,
        steps=steps,
    )
    runner.log_config_snapshot(build_snapshot(phase_config_path))

    # validate-only mode
    if cfg.validate_only:
        print("\n[validate-only] checking expected artifacts from the pipeline definition ...")
        all_ok = True
        for st in sorted(set(s.stage for s in steps)):
            stage_name = STAGE_NAMES.get(st, f"Stage {st}")
            print(f"\n  Stage {st} ({stage_name})")
            for step in [x for x in steps if x.stage == st]:
                if not step.expected_outputs:
                    print(f"    - {step.label}: (no expected outputs)")
                    continue
                ok, missing = validate_outputs(
                    output_root_native,
                    step.expected_outputs,
                    step.expected_min_bytes,
                    data_root_native=data_root_native,
                )
                if ok:
                    print(f"    OK  {step.label}")
                else:
                    all_ok = False
                    print(f"    FAIL  {step.label}")
                    for m in missing[:3]:
                        print(f"       - {m}")
                    if len(missing) > 3:
                        print(f"       ... (+{len(missing)-3} more)")
        return 0 if all_ok else 1

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
                    print("\n[STOP] halted due to a critical failure or --fail-fast.")
                    runner.print_summary()
                    return 1

    runner.print_summary()
    err_report = runner.write_error_code_report()
    print(f"Error code report: {err_report}")
    elapsed = time.time() - start
    print(f"\nTotal elapsed time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Contract check: Stage 5 (change-point) standard artifacts must exist after pipeline run.
    if 5 in planned_stages:
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
        print(f"Change-point artifact contract check: {cp_contract_path}")
        if not cp_ok:
            print("[ERROR] required change-point artifacts are missing.")
            for m in cp_missing:
                print(f"  - {m}")
            return 1

    # exit code
    if not ok_all:
        return 1

    # treat any NG as non-zero
    ng = [r for r in runner.records if r.status in ("FAIL", "FAIL_OUTPUT", "TIMEOUT", "ERROR", "NOT_FOUND")]
    if ng:
        print(f"\nWARNING: {len(ng)} failures detected. Log: {log_jsonl}")
        return 1

    return 0
