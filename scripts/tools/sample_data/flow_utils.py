from __future__ import annotations

import csv
import difflib
import hashlib
import io
import json
import os
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from scripts.tools.sample_data import ROOT, SRC

FLOW_TARGETS = {
    "manifest.csv": "manifest.normalized.csv",
    "merged_for_ai.md": "merged_for_ai.normalized.md",
    "merged_for_ai.zip": "merged_for_ai.zip.normalized.json",
}


@dataclass(frozen=True)
class FlowResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    output_dir: Path


def _path_variants(path: Path) -> list[str]:
    resolved = str(path.resolve())
    return sorted(
        {
            resolved,
            resolved.replace("\\", "/"),
            resolved.replace("/", "\\"),
        },
        key=len,
        reverse=True,
    )


def _normalize_path_field(raw_path: str, sample_root: Path, input_dir: Path, run_output_dir: Path) -> str:
    p = Path(raw_path)
    try:
        rp = p.resolve()
    except OSError:
        return raw_path.replace("\\", "/")

    for parent, token in ((input_dir, "<SAMPLE_INPUT>"), (sample_root, "<SAMPLE_ROOT>"), (run_output_dir, "<RUN_OUTPUT>")):
        try:
            rel = rp.relative_to(parent.resolve())
            return f"{token}/{rel.as_posix()}"
        except ValueError:
            continue
    return raw_path.replace("\\", "/")


def _normalize_text_common(text: str, sample_root: Path, input_dir: Path, run_output_dir: Path) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for value in _path_variants(input_dir):
        normalized = normalized.replace(value, "<SAMPLE_INPUT>")
    for value in _path_variants(sample_root):
        normalized = normalized.replace(value, "<SAMPLE_ROOT>")
    for value in _path_variants(run_output_dir):
        normalized = normalized.replace(value, "<RUN_OUTPUT>")
    return normalized


def normalize_manifest_csv(path: Path, sample_root: Path, input_dir: Path, run_output_dir: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])
    if "abs_path" in fieldnames:
        for row in rows:
            row["abs_path"] = _normalize_path_field(row.get("abs_path", ""), sample_root=sample_root, input_dir=input_dir, run_output_dir=run_output_dir)
    if "rel_path" in fieldnames:
        for row in rows:
            row["rel_path"] = row.get("rel_path", "").replace("\\", "/")
    rows.sort(key=lambda row: row.get("rel_path", ""))

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return out.getvalue().replace("\r\n", "\n")


def normalize_markdown(path: Path, sample_root: Path, input_dir: Path, run_output_dir: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return _normalize_text_common(raw, sample_root=sample_root, input_dir=input_dir, run_output_dir=run_output_dir)


def normalize_zip_summary(path: Path, sample_root: Path, input_dir: Path, run_output_dir: Path) -> str:
    entries: list[dict[str, object]] = []
    with zipfile.ZipFile(path, "r") as archive:
        for name in sorted(archive.namelist()):
            payload = archive.read(name)
            if name == "manifest.csv":
                text = normalize_manifest_csv_bytes(payload, sample_root=sample_root, input_dir=input_dir, run_output_dir=run_output_dir)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                entries.append({"name": name, "sha256": digest, "size": len(text.encode("utf-8"))})
                continue
            if name == "merged_for_ai.md":
                text = normalize_markdown_bytes(payload, sample_root=sample_root, input_dir=input_dir, run_output_dir=run_output_dir)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                entries.append({"name": name, "sha256": digest, "size": len(text.encode("utf-8"))})
                continue
            digest = hashlib.sha256(payload).hexdigest()
            entries.append({"name": name, "sha256": digest, "size": len(payload)})
    return json.dumps({"entries": entries}, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def normalize_manifest_csv_bytes(payload: bytes, sample_root: Path, input_dir: Path, run_output_dir: Path) -> str:
    text = payload.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])
    if "abs_path" in fieldnames:
        for row in rows:
            row["abs_path"] = _normalize_path_field(row.get("abs_path", ""), sample_root=sample_root, input_dir=input_dir, run_output_dir=run_output_dir)
    if "rel_path" in fieldnames:
        for row in rows:
            row["rel_path"] = row.get("rel_path", "").replace("\\", "/")
    rows.sort(key=lambda row: row.get("rel_path", ""))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return out.getvalue().replace("\r\n", "\n")


def normalize_markdown_bytes(payload: bytes, sample_root: Path, input_dir: Path, run_output_dir: Path) -> str:
    raw = payload.decode("utf-8", errors="replace")
    return _normalize_text_common(raw, sample_root=sample_root, input_dir=input_dir, run_output_dir=run_output_dir)


def collect_normalized_targets(flow_output_dir: Path, sample_root: Path, input_dir: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    run_output_dir = flow_output_dir.resolve()
    for source_name, target_name in FLOW_TARGETS.items():
        src = flow_output_dir / source_name
        if not src.exists():
            continue
        if source_name.endswith(".csv"):
            values[target_name] = normalize_manifest_csv(src, sample_root=sample_root, input_dir=input_dir, run_output_dir=run_output_dir)
        elif source_name.endswith(".md"):
            values[target_name] = normalize_markdown(src, sample_root=sample_root, input_dir=input_dir, run_output_dir=run_output_dir)
        elif source_name.endswith(".zip"):
            values[target_name] = normalize_zip_summary(src, sample_root=sample_root, input_dir=input_dir, run_output_dir=run_output_dir)
    return values


def run_artifact_legacy_flow(sample_input_dir: Path, flow_output_dir: Path, python_exe: str) -> FlowResult:
    flow_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        "-m",
        "arena.artifact_cli",
        "run",
        "--base",
        str(sample_input_dir.resolve()),
        "--out",
        str(flow_output_dir.resolve()),
        "--legacy",
        "--deterministic",
        "--no-ai-export",
    ]
    env = dict(os.environ)
    pypath = [str(SRC), str(ROOT)]
    if env.get("PYTHONPATH"):
        pypath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pypath)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    return FlowResult(command=cmd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, output_dir=flow_output_dir.resolve())


def unified_diff_snippet(expected: str, actual: str, expected_name: str, actual_name: str, max_lines: int = 80) -> str:
    diff_lines = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=expected_name,
            tofile=actual_name,
            lineterm="",
        )
    )
    if not diff_lines:
        return ""
    if len(diff_lines) > max_lines:
        omitted = len(diff_lines) - max_lines
        diff_lines = [*diff_lines[:max_lines], f"... diff truncated ({omitted} lines omitted)"]
    return "\n".join(diff_lines)
