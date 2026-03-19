from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.tools.sample_data import (
    DEFAULT_FIXED_MTIME_EPOCH,
    DEFAULT_GENERATION_TIMESTAMP,
    DEFAULT_MODE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_NAME,
    DEFAULT_SAMPLE_ROOT,
    DEFAULT_SAMPLE_TYPE,
    EXPECTED_DIRNAME,
    INPUT_DIRNAME,
    MANIFEST_NAME,
    SUPPORTED_MODES,
)
from scripts.tools.sample_data.manifest_schema import ensure_manifest
from scripts.tools.sample_data.sample_generators import build_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_public_sample",
        description="Build deterministic public smoke sample inputs for release-layer reproducibility.",
    )
    parser.add_argument("--sample-root", default=str(DEFAULT_SAMPLE_ROOT), help="Sample root directory.")
    parser.add_argument("--sample-name", default=DEFAULT_SAMPLE_NAME, help="Logical sample name written to manifest.")
    parser.add_argument("--sample-type", default=DEFAULT_SAMPLE_TYPE, help="Sample type label written to manifest.")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=DEFAULT_MODE, help="Sample build mode (future-compatible extension point).")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed used for deterministic synthetic generation.")
    parser.add_argument(
        "--generation-timestamp",
        default=DEFAULT_GENERATION_TIMESTAMP,
        help="Manifest generation timestamp label. Fixed by default for deterministic output.",
    )
    parser.add_argument("--fixed-mtime-epoch", type=int, default=DEFAULT_FIXED_MTIME_EPOCH, help="Fixed mtime (UNIX epoch seconds) applied to input files.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing sample root.")
    return parser


def _build_manifest_payload(
    *,
    sample_name: str,
    sample_type: str,
    source_type: str,
    generation_timestamp: str,
    random_seed: int,
    files_written: list[str],
    row_counts: dict[str, int],
    mode: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "sample_name": sample_name,
        "sample_type": sample_type,
        "source_type": source_type,
        "generation_timestamp": generation_timestamp,
        "random_seed": random_seed,
        "files_written": sorted(files_written),
        "row_counts": row_counts,
        "intended_usage": [
            "Release-layer reproducibility for command execution and artifact generation.",
            "Deterministic smoke checks for freeze/verify workflow in public repositories.",
        ],
        "compatibility_notes": [
            "This dataset is synthetic and intentionally lightweight.",
            "This dataset is not intended for scientific ADS-B performance conclusions.",
            "Mode extension placeholder: from-real cutout can be added later without changing CLI contract.",
        ],
        "mode": mode,
        "expected_outputs_written": [],
        "freeze_command_used": [],
        "freeze_timestamp": "",
    }
    ensure_manifest(payload)
    return payload


def run_from_args(args: argparse.Namespace) -> int:
    sample_root = Path(args.sample_root).resolve()
    input_dir = sample_root / INPUT_DIRNAME
    expected_dir = sample_root / EXPECTED_DIRNAME
    manifest_path = sample_root / MANIFEST_NAME

    if sample_root.exists() and not args.force:
        existing_content = list(sample_root.rglob("*"))
        if existing_content:
            print(f"[ERROR] sample root already exists and is not empty: {sample_root}", file=sys.stderr)
            print("        Re-run with --force to overwrite.", file=sys.stderr)
            return 2

    if sample_root.exists() and args.force:
        shutil.rmtree(sample_root)

    sample_root.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = build_sample(mode=args.mode, input_dir=input_dir, seed=args.seed, fixed_mtime_epoch=args.fixed_mtime_epoch)
    except NotImplementedError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    payload = _build_manifest_payload(
        sample_name=args.sample_name,
        sample_type=args.sample_type,
        source_type=result.source_type,
        generation_timestamp=args.generation_timestamp,
        random_seed=args.seed,
        files_written=result.files_written,
        row_counts=result.row_counts,
        mode=args.mode,
    )
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"[OK] sample_root: {sample_root}")
    print(f"[OK] input_dir: {input_dir}")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[SUMMARY] files_written={len(result.files_written)} total_rows={sum(result.row_counts.values())}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
