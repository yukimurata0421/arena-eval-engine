from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_repo_paths_on_syspath() -> None:
    # This CLI is intended to be used from a repo checkout where `scripts/` exists.
    # We resolve the repo root from the installed package location and add it to sys.path
    # so we can delegate to the legacy orchestration layer under `scripts/tools/`.
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    for candidate in (root, src):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="artifact", description="Artifact subsystem CLI")
    sub = p.add_subparsers(dest="subcommand", required=True)

    p_run = sub.add_parser("run", help="Export AI review artifact bundle from output/")
    p_run.add_argument(
        "--output-dir",
        default="",
        help="Convenience: set base/out relative to this output dir (base=<output-dir>, out=<output-dir>/payload) unless overridden",
    )
    p_run.add_argument("--base", default="", help="Base directory to scan (overrides --output-dir)")
    p_run.add_argument("--out", default="", help="Output directory (overrides --output-dir)")
    p_run.add_argument("--dry-run", action="store_true", help="Only create manifest (no merged file)")
    p_run.add_argument("--deterministic", action="store_true", help="Enable deterministic ordering/metadata where supported")
    p_run.add_argument("--no-ai-export", action="store_true", help="Disable AI bundle export stage (legacy mode only)")
    p_run.add_argument("--legacy", action="store_true", help="Use legacy flat output mode (manifest/merged/zip)")
    p_run.add_argument("--export-ai-folder", action="store_true", help="(Legacy) Export AI-selected files to a timestamped folder")
    p_run.add_argument("args", nargs=argparse.REMAINDER, help="Additional arguments forwarded to legacy runner")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.subcommand == "run":
        _ensure_repo_paths_on_syspath()
        from scripts.tools.artifacts.cli import main as legacy_main

        forwarded: list[str] = []

        output_dir = (Path(args.output_dir).resolve() if str(args.output_dir).strip() else None)
        base = (Path(args.base).resolve() if str(args.base).strip() else None) or output_dir
        out = (Path(args.out).resolve() if str(args.out).strip() else None) or (output_dir / "payload" if output_dir else None)

        if base is not None:
            forwarded += ["--base", str(base)]
        if out is not None:
            forwarded += ["--out", str(out)]

        if args.dry_run:
            forwarded.append("--dry-run")
        if args.deterministic:
            forwarded.append("--deterministic")
        if args.no_ai_export:
            forwarded.append("--no-ai-export")
        if args.legacy:
            forwarded.append("--legacy-flat-output")
        if args.export_ai_folder:
            forwarded.append("--export-ai-folder")

        passthrough = list(args.args)
        if passthrough[:1] == ["--"]:
            passthrough = passthrough[1:]
        forwarded += passthrough

        # Keep a small compatibility hook: also set env var for subprocesses that still look it up.
        if output_dir is not None:
            os.environ["ARENA_OUTPUT_DIR"] = str(output_dir)

        return int(legacy_main(forwarded))
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())

