from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.tools.sample_data import (
    DEFAULT_MODE,
    DEFAULT_SAMPLE_ROOT,
    EXPECTED_DIRNAME,
    INPUT_DIRNAME,
    MANIFEST_NAME,
    SUPPORTED_MODES,
)
from scripts.tools.sample_data.flow_utils import collect_normalized_targets, run_artifact_legacy_flow
from scripts.tools.sample_data.manifest_schema import ensure_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="freeze_expected_outputs",
        description="Run the public flow on smoke sample input and freeze deterministic expected outputs.",
    )
    parser.add_argument("--sample-root", default=str(DEFAULT_SAMPLE_ROOT), help="Sample root directory.")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=DEFAULT_MODE, help="Flow mode. from-real is reserved for future cutout support.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used to run artifact CLI.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing expected files.")
    parser.add_argument("--freeze-timestamp", default="", help="Explicit freeze timestamp (ISO-8601). If omitted, current UTC is used.")
    parser.add_argument("--keep-workdir", action="store_true", help="Keep the temporary freeze work directory for inspection.")
    return parser


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    ensure_manifest(payload)
    return payload


def _resolve_freeze_timestamp(raw: str) -> str:
    if raw.strip():
        return raw.strip()
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sanitize_command(command: list[str], input_dir: Path, work_dir: Path, python_exe: str) -> list[str]:
    normalized: list[str] = []
    python_resolved = str(Path(python_exe).resolve())
    input_resolved = str(input_dir.resolve())
    work_resolved = str(work_dir.resolve())
    for token in command:
        if token == python_resolved:
            normalized.append("<PYTHON_EXE>")
            continue
        if token == input_resolved:
            normalized.append("<SAMPLE_INPUT>")
            continue
        if token == work_resolved:
            normalized.append("<FREEZE_WORKDIR>")
            continue
        normalized.append(token)
    return normalized


def run_from_args(args: argparse.Namespace) -> int:
    sample_root = Path(args.sample_root).resolve()
    input_dir = sample_root / INPUT_DIRNAME
    expected_dir = sample_root / EXPECTED_DIRNAME
    manifest_path = sample_root / MANIFEST_NAME
    work_dir = sample_root / ".freeze_work"

    try:
        manifest = _load_manifest(manifest_path)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[ERROR] failed to load manifest: {exc}", file=sys.stderr)
        return 2

    if args.mode != str(manifest.get("mode", "")):
        print(
            f"[ERROR] mode mismatch: manifest mode={manifest.get('mode')} requested mode={args.mode}.",
            file=sys.stderr,
        )
        return 2

    if not input_dir.exists():
        print(f"[ERROR] sample input directory not found: {input_dir}", file=sys.stderr)
        return 2

    if expected_dir.exists():
        existing = [p for p in expected_dir.iterdir()]
        if existing and not args.force:
            print(f"[ERROR] expected directory is not empty: {expected_dir}", file=sys.stderr)
            print("        Re-run with --force to overwrite.", file=sys.stderr)
            return 2
        if existing and args.force:
            shutil.rmtree(expected_dir)
    expected_dir.mkdir(parents=True, exist_ok=True)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    flow = run_artifact_legacy_flow(sample_input_dir=input_dir, flow_output_dir=work_dir, python_exe=args.python_exe)
    if flow.returncode != 0:
        print("[ERROR] artifact flow execution failed.", file=sys.stderr)
        print(f"        command: {' '.join(flow.command)}", file=sys.stderr)
        if flow.stdout.strip():
            print("        stdout tail:", file=sys.stderr)
            print(flow.stdout.strip()[-1200:], file=sys.stderr)
        if flow.stderr.strip():
            print("        stderr tail:", file=sys.stderr)
            print(flow.stderr.strip()[-1200:], file=sys.stderr)
        return int(flow.returncode) if int(flow.returncode) != 0 else 1

    normalized_targets = collect_normalized_targets(flow.output_dir, sample_root=sample_root, input_dir=input_dir)
    if not normalized_targets:
        print("[ERROR] no flow outputs were collected for freezing.", file=sys.stderr)
        return 1

    for name, content in normalized_targets.items():
        (expected_dir / name).write_text(content, encoding="utf-8", newline="\n")

    manifest["expected_outputs_written"] = sorted(normalized_targets.keys())
    manifest["freeze_command_used"] = _sanitize_command(flow.command, input_dir=input_dir, work_dir=work_dir, python_exe=args.python_exe)
    manifest["freeze_timestamp"] = _resolve_freeze_timestamp(args.freeze_timestamp)
    ensure_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    if not args.keep_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"[OK] expected_dir: {expected_dir}")
    print(f"[OK] manifest updated: {manifest_path}")
    print(f"[SUMMARY] expected_outputs_written={len(normalized_targets)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
