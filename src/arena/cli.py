from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from arena.log import get_logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena import pipeline
from arena.artifacts.integrity import verify_artifact_bundle
from arena.artifacts.replay import replay_artifact_bundle
from arena.lib.config_resolution import build_runtime_config_metadata, validate_resolved_config_paths
from arena.lib.paths import resolve_data_dir, resolve_output_dir, resolve_root, resolve_scripts_root
from arena.lib.runtime_config import clear_settings_cache, load_settings

logger = get_logger(__name__)


def _apply_path_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "scripts_root", None):
        os.environ["ARENA_SCRIPTS_ROOT"] = str(Path(args.scripts_root).resolve())
    if getattr(args, "data_dir", None):
        os.environ["ARENA_DATA_DIR"] = str(Path(args.data_dir).resolve())
    if getattr(args, "output_dir", None):
        os.environ["ARENA_OUTPUT_DIR"] = str(Path(args.output_dir).resolve())
    if getattr(args, "settings", None):
        os.environ["ARENA_SETTINGS"] = str(Path(args.settings).resolve())
    if getattr(args, "phase_config", None):
        os.environ["ARENA_PHASE_CONFIG"] = str(Path(args.phase_config).resolve())
    if getattr(args, "analysis_start_date", None):
        os.environ["ARENA_ANALYSIS_START_DATE"] = str(args.analysis_start_date).strip()
    if getattr(args, "analysis_end_date", None):
        os.environ["ARENA_ANALYSIS_END_DATE"] = str(args.analysis_end_date).strip()


def _resolve_config_metadata(args: argparse.Namespace) -> dict[str, object]:
    scripts_root = str(Path(args.scripts_root).resolve()) if getattr(args, "scripts_root", None) else os.getenv("ARENA_SCRIPTS_ROOT", "")
    settings_override = str(Path(args.settings).resolve()) if getattr(args, "settings", None) else None
    phase_override = str(Path(args.phase_config).resolve()) if getattr(args, "phase_config", None) else None
    metadata = build_runtime_config_metadata(
        settings_override=settings_override,
        phase_override=phase_override,
        scripts_root=scripts_root or None,
        analysis_start_date=str(getattr(args, "analysis_start_date", "") or ""),
        analysis_end_date=str(getattr(args, "analysis_end_date", "") or ""),
        env={},
    )
    os.environ["ARENA_SETTINGS"] = str(metadata["resolved_settings_path"])
    os.environ["ARENA_PHASE_CONFIG"] = str(metadata["resolved_phase_config_path"])
    os.environ["ADSB_SETTINGS"] = str(metadata["resolved_settings_path"])
    os.environ["ADSB_PHASE_CONFIG"] = str(metadata["resolved_phase_config_path"])
    clear_settings_cache()
    return metadata


def cmd_validate(args: argparse.Namespace) -> int:
    _apply_path_overrides(args)
    config_meta = _resolve_config_metadata(args)
    config_errors = validate_resolved_config_paths(config_meta)

    settings_snapshot = load_settings(force_reload=True)
    settings_path = Path(settings_snapshot.path)
    phase_cfg = Path(str(config_meta["resolved_phase_config_path"]))

    default_scripts_root = resolve_scripts_root()
    default_root = resolve_root(scripts_root=default_scripts_root)
    scripts_root = Path(os.getenv("ARENA_SCRIPTS_ROOT", str(default_scripts_root)))
    data_dir = Path(os.getenv("ARENA_DATA_DIR", str(resolve_data_dir(root=default_root))))
    output_dir = Path(os.getenv("ARENA_OUTPUT_DIR", str(resolve_output_dir(root=default_root))))

    ok = True
    if not scripts_root.exists():
        logger.error("[NG] scripts root not found: %s", scripts_root)
        ok = False
    if not (scripts_root / "adsb").exists():
        logger.error("[NG] scripts/adsb not found: %s", scripts_root / "adsb")
        ok = False

    logger.info("resolved_settings_path: %s", config_meta["resolved_settings_path"])
    logger.info("resolved_phase_config_path: %s", config_meta["resolved_phase_config_path"])
    logger.info("used_default_settings: %d", int(bool(config_meta["used_default_settings"])))
    logger.info("used_default_phase_config: %d", int(bool(config_meta["used_default_phase_config"])))
    logger.info("experimental_mode: %d", int(bool(config_meta["experimental_mode"])))
    if str(config_meta.get("analysis_start_date", "")):
        logger.info("analysis_start_date: %s", config_meta.get("analysis_start_date"))
    if str(config_meta.get("analysis_end_date", "")):
        logger.info("analysis_end_date: %s", config_meta.get("analysis_end_date"))

    if config_errors:
        ok = False
        for err in config_errors:
            logger.error("[NG] %s", err)

    if not settings_path.exists():
        logger.error("[NG] settings.toml not found: %s", settings_path)
        ok = False
    else:
        logger.info("[OK] settings.toml: %s", settings_path)
        data = settings_snapshot.data or {}
        site_ok = "site" in data and isinstance(data["site"], dict) and {"lat", "lon"}.issubset(set(data["site"].keys()))
        quality_ok = (
            "quality" in data
            and isinstance(data["quality"], dict)
            and {"min_auc_n_used", "min_minutes_covered"}.issubset(set(data["quality"].keys()))
        )
        bins_ok = "distance_bins" in data and isinstance(data["distance_bins"], dict) and "km" in data["distance_bins"]
        if not (site_ok and quality_ok and bins_ok):
            logger.error("[NG] settings.toml is missing a required key (site/quality/distance_bins)")
            ok = False
        else:
            try:
                lat = float(data["site"]["lat"])
                lon = float(data["site"]["lon"])
                if lat == 0.0 and lon == 0.0:
                    logger.warning("Warning: site.lat/lon is 0.0 (not set). Please set [site] in settings.toml.")
            except (TypeError, ValueError):
                logger.error("[NG] lat/lon in settings.toml [site] must be a number")
                ok = False

    if phase_cfg and phase_cfg.exists():
        logger.info("[OK] phases.txt: %s", phase_cfg)
    else:
        logger.error("[NG] phases.txt not found (set --phase-config or ARENA_PHASE_CONFIG)")
        ok = False

    if args.create_dirs:
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        logger.error("[NG] data directory not found: %s", data_dir)
        ok = False
    else:
        logger.info("[OK] data dir: %s", data_dir)

    if not output_dir.exists():
        logger.error("[NG] output directory not found: %s", output_dir)
        ok = False
    else:
        logger.info("[OK] output dir: %s", output_dir)

    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    _apply_path_overrides(args)
    config_meta = _resolve_config_metadata(args)
    config_errors = validate_resolved_config_paths(config_meta)
    if config_errors:
        for err in config_errors:
            logger.error("[NG] %s", err)
        return 1

    cfg = pipeline.RunConfig(
        stage=args.stage,
        only=args.only,
        dry_run=args.dry_run,
        no_gpu=args.no_gpu,
        full=args.full,
        backend=args.backend,
        scripts_root=str(Path(args.scripts_root).resolve()) if args.scripts_root else "",
        output_root=str(Path(args.output_dir).resolve()) if args.output_dir else "",
        data_root=str(Path(args.data_dir).resolve()) if args.data_dir else "",
        dynamic_date=args.dynamic_date or "",
        phase_config=str(config_meta["resolved_phase_config_path"]),
        settings_path=str(config_meta["resolved_settings_path"]),
        validate=not args.no_validate,
        validate_only=args.validate_only,
        skip_existing=args.skip_existing,
        fail_fast=args.fail_fast,
        log_jsonl=args.log_jsonl or "",
        skip_plao=args.skip_plao,
        workers=args.workers,
    )

    return pipeline.run(cfg)


def cmd_fetch_opensky(args: argparse.Namespace) -> int:
    _apply_path_overrides(args)

    script = Path(os.getenv("ARENA_SCRIPTS_ROOT", str(resolve_scripts_root()))) / "adsb" / "data_fetch" / "get_opensky_traffic.py"
    if not script.exists():
        logger.error("[NG] OpenSky script not found: %s", script)
        return 1

    env = {**os.environ}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    cmd = [sys.executable, str(script)]
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def cmd_sync_rpi_logs(args: argparse.Namespace) -> int:
    _apply_path_overrides(args)

    script = Path(os.getenv("ARENA_SCRIPTS_ROOT", str(resolve_scripts_root()))) / "adsb" / "ops" / "rpi_log_sync.py"
    if not script.exists():
        logger.error("[NG] RPi sync script not found: %s", script)
        return 1

    env = {**os.environ}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    cmd = [sys.executable, str(script)]
    if args.host:
        cmd += ["--host", args.host]
    if args.user:
        cmd += ["--user", args.user]
    if args.port:
        cmd += ["--port", str(args.port)]
    if args.remote_dir:
        cmd += ["--remote-dir", args.remote_dir]
    if args.plao_remote_dir:
        cmd += ["--plao-remote-dir", args.plao_remote_dir]
    if args.plao_local_dir:
        cmd += ["--plao-local-dir", args.plao_local_dir]
    if args.ssh_key:
        cmd += ["--ssh-key", args.ssh_key]
    if args.strict_host_key_checking:
        cmd += ["--strict-host-key-checking", args.strict_host_key_checking]
    if args.output_json:
        cmd += ["--output-json", args.output_json]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.fail_on_missing_remote:
        cmd.append("--fail-on-missing-remote")
    if args.skip_plao_sync:
        cmd.append("--skip-plao-sync")

    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def cmd_artifacts_verify(args: argparse.Namespace) -> int:
    try:
        result = verify_artifact_bundle(Path(args.artifact_bundle))
    except (OSError, ValueError, KeyError) as exc:
        logger.error("[NG] artifact verification failed: %s", exc)
        return 1
    print(f"artifact_bundle: {Path(args.artifact_bundle).resolve()}")
    print(f"valid: {int(bool(result['valid']))}")
    print(f"bundle_sha256: {result['bundle_sha256']}")
    integrity_summary = result.get("integrity_summary", {})
    if isinstance(integrity_summary, dict):
        print(f"integrity_passed: {int(bool(integrity_summary.get('passed', False)))}")
    errors = result.get("errors", [])
    if isinstance(errors, list) and errors:
        print("errors:")
        for error in errors:
            print(f"- {error}")
    return 0 if result["valid"] else 1


def cmd_artifacts_replay(args: argparse.Namespace) -> int:
    try:
        return replay_artifact_bundle(Path(args.artifact_bundle))
    except (OSError, ValueError, KeyError) as exc:
        logger.error("[NG] artifact replay failed: %s", exc)
        return 1


def _expand_synthesis_shorthand(forwarded: list[str]) -> list[str]:
    if not forwarded:
        return forwarded

    # Daily-operation shorthand:
    # arena synthesis enrich            -> enrich --only-unreviewed
    # arena synthesis cluster-baselines -> cluster-baselines --rebuild
    # arena synthesis report-baselines  -> report-baselines --limit 20
    # arena synthesis suggest-actions   -> suggest-actions --min-severity high --sort-by score
    if forwarded == ["enrich"]:
        return ["enrich", "--only-unreviewed"]
    if forwarded == ["cluster-baselines"]:
        return ["cluster-baselines", "--rebuild"]
    if forwarded == ["report-baselines"]:
        return ["report-baselines", "--limit", "20"]
    if forwarded == ["suggest-actions"]:
        return ["suggest-actions", "--min-severity", "high", "--sort-by", "score"]
    return forwarded


def cmd_synthesis(args: argparse.Namespace) -> int:
    from arena.synthesis import cli as synthesis_cli

    forwarded = list(args.synthesis_args or [])
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if not forwarded:
        forwarded = ["--help"]
    forwarded = _expand_synthesis_shorthand(forwarded)
    return int(synthesis_cli.main(forwarded))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arena", description="ARENA Rating Engine CLI")
    sub = p.add_subparsers(dest="subcommand", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scripts-root", help="Override scripts root")
    common.add_argument("--data-dir", help="Overwrite data directory")
    common.add_argument("--output-dir", help="Overwrite output directory")
    common.add_argument("--settings", "--config", dest="settings", help="settings.toml path")
    common.add_argument("--phase-config", help="phases.txt path")
    common.add_argument("--analysis-start-date", help="Start date of analysis target (YYYY-MM-DD). If not specified, whole period.")
    common.add_argument("--analysis-end-date", help="End date of analysis target (YYYY-MM-DD). If not specified, whole period.")

    p_run = sub.add_parser("run", parents=[common], help="Run evaluation pipeline")
    p_run.add_argument("--stage", type=int, default=1)
    p_run.add_argument("--only", type=int, default=None)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--no-gpu", action="store_true")
    p_run.add_argument("--full", action="store_true")
    p_run.add_argument("--backend", choices=["auto", "native", "wsl"], default="auto")
    p_run.add_argument("--dynamic-date", default="")
    p_run.add_argument("--no-validate", action="store_true")
    p_run.add_argument("--validate-only", action="store_true")
    p_run.add_argument("--skip-existing", action="store_true")
    p_run.add_argument("--fail-fast", action="store_true")
    p_run.add_argument("--log-jsonl", default="")
    p_run.add_argument("--skip-plao", action="store_true")
    p_run.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallelism degree (0=auto: number of logical CPUs). Used for step parallelism in stage 2/3/5 and the number of ProcessPool/ThreadPool/Bayesian chains in each script",
    )
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser("validate", parents=[common], help="Validate settings and directories")
    p_val.add_argument("--create-dirs", action="store_true", help="Create data/output if it does not exist")
    p_val.set_defaults(func=cmd_validate)

    p_fetch = sub.add_parser("fetch-opensky", parents=[common], help="Fetch OpenSky traffic data")
    p_fetch.set_defaults(func=cmd_fetch_opensky)

    p_sync = sub.add_parser("sync-rpi-logs", parents=[common], help="Differential synchronization of Raspberry Pi logs")
    p_sync.add_argument("--host", default="", help="Raspberry Pi hostname/IP")
    p_sync.add_argument("--user", default="", help="SSH user")
    p_sync.add_argument("--port", type=int, default=0, help="SSH port")
    p_sync.add_argument("--remote-dir", default="", help="Remote log directory")
    p_sync.add_argument("--plao-remote-dir", default="", help="Remote plao_pos directory")
    p_sync.add_argument("--plao-local-dir", default="", help="Local plao_pos directory")
    p_sync.add_argument("--ssh-key", default="", help="SSH private key path")
    p_sync.add_argument(
        "--strict-host-key-checking",
        choices=["accept-new", "yes", "no"],
        default="",
        help="SSH StrictHostKeyChecking policy",
    )
    p_sync.add_argument("--output-json", default="", help="Sync report JSON path")
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.add_argument("--fail-on-missing-remote", action="store_true")
    p_sync.add_argument("--skip-plao-sync", action="store_true")
    p_sync.set_defaults(func=cmd_sync_rpi_logs)

    p_synthesis = sub.add_parser(
        "synthesis",
        aliases=["distill"],
        help="Synthesis CLI (ingest/enrich/proposition/review helpers with daily shorthand defaults)",
    )
    p_synthesis.add_argument("synthesis_args", nargs=argparse.REMAINDER, help="Arguments passed through to synthesis CLI")
    p_synthesis.set_defaults(func=cmd_synthesis)

    p_artifacts = sub.add_parser("artifacts", help="Validating and reevaluating artifact bundles")
    artifacts_sub = p_artifacts.add_subparsers(dest="artifacts_command", required=True)

    p_artifacts_verify = artifacts_sub.add_parser("verify", help="Verify artifact bundle")
    p_artifacts_verify.add_argument("artifact_bundle", nargs="?", default=".", help="artifact bundle root")
    p_artifacts_verify.set_defaults(func=cmd_artifacts_verify)

    p_artifacts_replay = artifacts_sub.add_parser("replay", help="Re-evaluate artifact bundle")
    p_artifacts_replay.add_argument("artifact_bundle", help="artifact bundle root")
    p_artifacts_replay.set_defaults(func=cmd_artifacts_replay)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
