from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

def _bootstrap_repo_imports() -> None:
    for candidate in (ROOT, SRC):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _load_runtime_defaults() -> tuple[Path, tuple[str, ...]]:
    _bootstrap_repo_imports()
    from arena.artifacts.policies import TEXT_EXT_DEFAULT
    from arena.lib.paths import resolve_output_dir

    return resolve_output_dir(), tuple(TEXT_EXT_DEFAULT)


def build_parser() -> argparse.ArgumentParser:
    output_dir, text_ext_default = _load_runtime_defaults()
    parser = argparse.ArgumentParser()
    default_base = str(output_dir)
    default_out = str(output_dir / "merged_for_ai")
    parser.add_argument("--base", default=default_base, help="Base directory to scan")
    parser.add_argument("--out", default=default_out, help="Output directory")
    parser.add_argument(
        "--include-ext",
        default=",".join(text_ext_default),
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
        help="Export only AI-selected files to a timestamped folder under output dir (no zip)",
    )
    parser.add_argument("--no-ai-export", action="store_true", help="Disable default AI export in normal mode")
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


def main() -> int:
    _bootstrap_repo_imports()
    from scripts.tools.artifacts.app import run_from_args

    parser = build_parser()
    args = parser.parse_args()
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
