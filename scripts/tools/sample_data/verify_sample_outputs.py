from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.tools.sample_data import DEFAULT_MODE, DEFAULT_SAMPLE_ROOT, EXPECTED_DIRNAME, INPUT_DIRNAME, MANIFEST_NAME, SUPPORTED_MODES
from scripts.tools.sample_data.flow_utils import collect_normalized_targets, run_artifact_legacy_flow, unified_diff_snippet
from scripts.tools.sample_data.manifest_schema import ensure_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_sample_outputs",
        description="Run the public flow on smoke sample input and verify against frozen expected outputs.",
    )
    parser.add_argument("--sample-root", default=str(DEFAULT_SAMPLE_ROOT), help="Sample root directory.")
    parser.add_argument(
        "--expected-dir",
        default="",
        help="Optional expected outputs directory override. Defaults to <sample-root>/expected.",
    )
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=DEFAULT_MODE, help="Flow mode. from-real is reserved for future cutout support.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used to run artifact CLI.")
    parser.add_argument("--work-dir", default="", help="Optional fixed work directory for generated flow outputs.")
    parser.add_argument("--keep-workdir", action="store_true", help="Keep work directory after verification.")
    return parser


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    ensure_manifest(payload)
    return payload


def _load_expected(expected_dir: Path, expected_names: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in expected_names:
        path = expected_dir / name
        if path.exists():
            values[name] = path.read_text(encoding="utf-8", errors="replace")
    return values


def run_from_args(args: argparse.Namespace) -> int:
    sample_root = Path(args.sample_root).resolve()
    input_dir = sample_root / INPUT_DIRNAME
    expected_dir = Path(args.expected_dir).resolve() if args.expected_dir.strip() else sample_root / EXPECTED_DIRNAME
    manifest_path = sample_root / MANIFEST_NAME

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
    if not expected_dir.exists():
        print(f"[ERROR] expected directory not found: {expected_dir}", file=sys.stderr)
        return 2

    expected_names = sorted(str(x) for x in manifest.get("expected_outputs_written", []))
    if not expected_names:
        if args.expected_dir.strip():
            expected_names = sorted(path.name for path in expected_dir.iterdir() if path.is_file())
            if not expected_names:
                print(f"[ERROR] expected directory is empty: {expected_dir}", file=sys.stderr)
                return 2
            print("[INFO] manifest.expected_outputs_written is empty; using --expected-dir file list.")
        else:
            print("[ERROR] manifest.expected_outputs_written is empty. Run freeze_expected_outputs first.", file=sys.stderr)
            return 2

    remove_workdir = False
    if args.work_dir.strip():
        work_dir = Path(args.work_dir).resolve()
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="arena_sample_verify_", dir=str(sample_root)))
        remove_workdir = True

    flow = run_artifact_legacy_flow(sample_input_dir=input_dir, flow_output_dir=work_dir, python_exe=args.python_exe)
    if flow.returncode != 0:
        print("[ERROR] artifact flow execution failed during verification.", file=sys.stderr)
        print(f"        command: {' '.join(flow.command)}", file=sys.stderr)
        if flow.stderr.strip():
            print("        stderr tail:", file=sys.stderr)
            print(flow.stderr.strip()[-1200:], file=sys.stderr)
        if remove_workdir and not args.keep_workdir:
            shutil.rmtree(work_dir, ignore_errors=True)
        return int(flow.returncode) if int(flow.returncode) != 0 else 1

    actual_values = collect_normalized_targets(flow.output_dir, sample_root=sample_root, input_dir=input_dir)
    expected_values = _load_expected(expected_dir, expected_names)

    expected_set = set(expected_names)
    actual_set = set(actual_values.keys())
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)

    diffs: list[tuple[str, str]] = []
    matched: list[str] = []
    for name in sorted(expected_set & actual_set):
        expected_content = expected_values.get(name, "")
        actual_content = actual_values.get(name, "")
        if expected_content == actual_content:
            matched.append(name)
            continue
        diff = unified_diff_snippet(expected_content, actual_content, f"expected/{name}", f"actual/{name}")
        diffs.append((name, diff))

    print(f"[SUMMARY] matched={len(matched)} missing={len(missing)} unexpected={len(unexpected)} different={len(diffs)}")
    if matched:
        print("matched files:")
        for name in matched:
            print(f"- {name}")
    if missing:
        print("missing files:")
        for name in missing:
            print(f"- {name}")
    if unexpected:
        print("unexpected files:")
        for name in unexpected:
            print(f"- {name}")
    if diffs:
        print("content differences:")
        for name, diff in diffs:
            print(f"--- {name} ---")
            print(diff or "(binary difference)")

    ok = not missing and not unexpected and not diffs

    if remove_workdir and not args.keep_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)

    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
