from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from arena import pipeline
from arena.artifacts.integrity import verify_artifact_bundle
from arena.artifacts.replay import replay_artifact_bundle
from arena.lib.paths import resolve_data_dir, resolve_output_dir, resolve_root, resolve_scripts_root
from arena.lib.phase_config import load_phase_config
from arena.lib.runtime_config import load_settings


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


def cmd_validate(args: argparse.Namespace) -> int:
    _apply_path_overrides(args)

    settings_snapshot = load_settings()
    settings_path = Path(settings_snapshot.path)
    phase_cfg_env = os.getenv("ARENA_PHASE_CONFIG", "")
    if phase_cfg_env:
        phase_cfg = Path(phase_cfg_env)
    else:
        try:
            phase_cfg = Path(load_phase_config().config_path)
        except Exception:
            phase_cfg = Path()

    default_scripts_root = resolve_scripts_root()
    default_root = resolve_root(scripts_root=default_scripts_root)
    scripts_root = Path(os.getenv("ARENA_SCRIPTS_ROOT", str(default_scripts_root)))
    data_dir = Path(os.getenv("ARENA_DATA_DIR", str(resolve_data_dir(root=default_root))))
    output_dir = Path(os.getenv("ARENA_OUTPUT_DIR", str(resolve_output_dir(root=default_root))))

    ok = True
    if not scripts_root.exists():
        print(f"[NG] scripts root not found: {scripts_root}")
        ok = False
    if not (scripts_root / "adsb").exists():
        print(f"[NG] scripts/adsb not found: {scripts_root / 'adsb'}")
        ok = False

    if not settings_path.exists():
        print(f"[NG] settings.toml not found: {settings_path}")
        ok = False
    else:
        print(f"[OK] settings.toml: {settings_path}")
        data = settings_snapshot.data or {}
        site_ok = (
            "site" in data
            and isinstance(data["site"], dict)
            and {"lat", "lon"}.issubset(set(data["site"].keys()))
        )
        quality_ok = (
            "quality" in data
            and isinstance(data["quality"], dict)
            and {"min_auc_n_used", "min_minutes_covered"}.issubset(set(data["quality"].keys()))
        )
        bins_ok = (
            "distance_bins" in data
            and isinstance(data["distance_bins"], dict)
            and "km" in data["distance_bins"]
        )
        if not (site_ok and quality_ok and bins_ok):
            print("[NG] settings.toml missing required keys (site/quality/distance_bins)")
            ok = False
        else:
            try:
                lat = float(data["site"]["lat"])
                lon = float(data["site"]["lon"])
                if lat == 0.0 and lon == 0.0:
                    print("WARNING: site.lat/lon is 0.0 (not configured). Set it in settings.toml [site].")
            except (TypeError, ValueError):
                print("[NG] settings.toml [site] lat/lon must be numeric")
                ok = False

    if phase_cfg and phase_cfg.exists():
        print(f"[OK] phases.txt: {phase_cfg}")
    else:
        print("[NG] phases.txt not found (set --phase-config or ARENA_PHASE_CONFIG)")
        ok = False

    if args.create_dirs:
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"[NG] data dir not found: {data_dir}")
        ok = False
    else:
        print(f"[OK] data dir: {data_dir}")

    if not output_dir.exists():
        print(f"[NG] output dir not found: {output_dir}")
        ok = False
    else:
        print(f"[OK] output dir: {output_dir}")

    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    _apply_path_overrides(args)

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
        phase_config=str(Path(args.phase_config).resolve()) if args.phase_config else "",
        validate=not args.no_validate,
        validate_only=args.validate_only,
        skip_existing=args.skip_existing,
        fail_fast=args.fail_fast,
        log_jsonl=args.log_jsonl or "",
        log_jsonl_mode=args.log_jsonl_mode,
        skip_plao=args.skip_plao,
        workers=args.workers,
    )

    return pipeline.run(cfg)


def cmd_fetch_opensky(args: argparse.Namespace) -> int:
    _apply_path_overrides(args)

    script = Path(os.getenv("ARENA_SCRIPTS_ROOT", str(resolve_scripts_root()))) / "adsb" / "data_fetch" / "get_opensky_traffic.py"
    if not script.exists():
        print(f"[NG] OpenSky script not found: {script}")
        return 1

    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    proc = subprocess.run([sys.executable, str(script)], env=env)
    return proc.returncode


def cmd_artifacts_verify(args: argparse.Namespace) -> int:
    try:
        result = verify_artifact_bundle(Path(args.artifact_bundle))
    except Exception as exc:
        print(f"[NG] artifact verification failed: {exc}")
        return 1
    print(f"artifact_bundle: {Path(args.artifact_bundle).resolve()}")
    print(f"valid: {int(result['valid'])}")
    print(f"bundle_sha256: {result['bundle_sha256']}")
    print(f"integrity_passed: {int(bool(result['integrity_summary'].get('passed', False)))}")
    if result["errors"]:
        print("errors:")
        for error in result["errors"]:
            print(f"- {error}")
    return 0 if result["valid"] else 1


def cmd_artifacts_replay(args: argparse.Namespace) -> int:
    try:
        return replay_artifact_bundle(Path(args.artifact_bundle))
    except Exception as exc:
        print(f"[NG] artifact replay failed: {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arena", description="ARENA evaluation engine CLI")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scripts-root", help="override scripts root")
    common.add_argument("--data-dir", help="override data directory")
    common.add_argument("--output-dir", help="override output directory")
    common.add_argument("--settings", "--config", dest="settings", help="settings.toml path")
    common.add_argument("--phase-config", help="phases.txt path")

    p_run = sub.add_parser("run", parents=[common], help="run evaluation pipeline")
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
    p_run.add_argument("--log-jsonl-mode", choices=["append", "overwrite"], default="append")
    p_run.add_argument("--skip-plao", action="store_true")
    p_run.add_argument("--workers", type=int, default=0, help="worker count for child scripts (0=auto)")
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser("validate", parents=[common], help="validate config and directories")
    p_val.add_argument("--create-dirs", action="store_true", help="create data/output if missing")
    p_val.set_defaults(func=cmd_validate)

    p_fetch = sub.add_parser("fetch-opensky", parents=[common], help="fetch OpenSky traffic data")
    p_fetch.set_defaults(func=cmd_fetch_opensky)

    p_artifacts = sub.add_parser("artifacts", help="verify or replay an artifact bundle")
    artifacts_sub = p_artifacts.add_subparsers(dest="artifacts_command", required=True)

    p_artifacts_verify = artifacts_sub.add_parser("verify", help="verify an artifact bundle")
    p_artifacts_verify.add_argument("artifact_bundle", nargs="?", default=".", help="artifact bundle root")
    p_artifacts_verify.set_defaults(func=cmd_artifacts_verify)

    p_artifacts_replay = artifacts_sub.add_parser("replay", help="replay artifact bundle verification")
    p_artifacts_replay.add_argument("artifact_bundle", help="artifact bundle root")
    p_artifacts_replay.set_defaults(func=cmd_artifacts_replay)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
