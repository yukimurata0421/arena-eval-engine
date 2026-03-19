from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from arena.artifacts.policies import TEXT_EXT_DEFAULT
from arena.lib.paths import OUTPUT_DIR
from scripts.tools.artifacts.app import run_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="artifact run")
    default_base = str(OUTPUT_DIR)
    default_out = str(Path(OUTPUT_DIR) / "payload")
    parser.add_argument("--base", default=default_base, help="Base directory to scan")
    parser.add_argument("--out", default=default_out, help="Output directory")
    parser.add_argument(
        "--include-ext",
        default=",".join(TEXT_EXT_DEFAULT),
        help="Comma-separated extensions to include (e.g. .txt,.log,.json,.csv,.html)",
    )
    parser.add_argument(
        "--exclude-dir",
        default="",
        help="Comma-separated directory names to exclude in addition to always-excluded dirs",
    )
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=2_000_000,
        help="Max bytes to read per file (default: 2,000,000)",
    )
    parser.add_argument("--sort", choices=["path", "mtime"], default="path", help="Sort order for merged output")
    parser.add_argument("--dry-run", action="store_true", help="Only create manifest (no merged file)")
    parser.add_argument(
        "--export-ai-folder",
        action="store_true",
        help="(Legacy mode) Export only AI-selected files to a timestamped folder under output dir.",
    )
    parser.add_argument("--no-ai-export", action="store_true", help="Disable default AI export in normal mode")
    parser.add_argument(
        "--legacy-flat-output",
        action="store_true",
        help="Enable legacy merged_for_ai output generation (manifest.csv/merged_for_ai.md/merged_for_ai.zip).",
    )
    parser.add_argument(
        "--ai-export-root",
        default="",
        help="Root dir for --export-ai-folder. If omitted, use --out.",
    )
    parser.add_argument(
        "--ai-export-use-out-parent",
        action="store_true",
        help="(Deprecated) Kept for backward compatibility. Destination is still --out unless --ai-export-root is set.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic export metadata and ordering where supported.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
