# -*- coding: utf-8 -*-
"""
merge_output_for_ai.py

Concatenate text artifacts under output/ into a single Markdown file while
preserving directory and file names.

Primary use:
- Make WSL-generated output easy to ingest into AI/RAG pipelines.

Outputs:
- <out_dir>/manifest.csv
- <out_dir>/merged_for_ai.md
- <out_dir>/merged_for_ai.zip

Example:
  python merge_output_for_ai.py --base "<project>/output" --out "<project>/output/merged_for_ai"
"""


from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from arena.lib.paths import OUTPUT_DIR


TEXT_EXT_DEFAULT = [".txt", ".log", ".json", ".jsonl", ".csv", ".html", ".md"]


@dataclass
class FileItem:
    rel_path: str
    abs_path: str
    ext: str
    size_bytes: int
    mtime_iso: str
    included: bool
    reason: str


def iso_mtime(p: Path) -> str:
    ts = p.stat().st_mtime
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def try_read_text(path: Path, max_bytes: int) -> Tuple[Optional[str], str]:
    """
    Try to read as text.
    - UTF-8 (with BOM) -> fallback to cp932 -> final fallback latin-1
    - If size exceeds max_bytes, read the head and append a truncation note
    Returns: (text or None, encoding_used_or_error)
    """
    size = path.stat().st_size
    to_read = min(size, max_bytes)

    with path.open("rb") as f:
        head = f.read(min(to_read, 4096))
        if b"\x00" in head:
            return None, "binary_like(NULL found)"

    raw = None
    with path.open("rb") as f:
        raw = f.read(to_read)

    encodings = ["utf-8-sig", "utf-8", "cp932", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            text = raw.decode(enc, errors="strict")
            note = enc
            break
        except Exception as e:
            last_err = e
            text = None
            note = None

    if text is None:
        return None, f"decode_failed({last_err})"

    if size > max_bytes:
        text += "\n\n<!-- TRUNCATED: file_size={} bytes exceeded max_bytes={} -->\n".format(size, max_bytes)

    return text, note


def enumerate_files(base_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in base_dir.rglob("*"):
        if p.is_file():
            files.append(p)
    return files


def normalize_rel(base_dir: Path, p: Path) -> str:
    rel = p.relative_to(base_dir).as_posix()
    return rel


def write_manifest(items: List[FileItem], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["included", "reason", "rel_path", "abs_path", "ext", "size_bytes", "mtime_iso"])
        for it in items:
            w.writerow([int(it.included), it.reason, it.rel_path, it.abs_path, it.ext, it.size_bytes, it.mtime_iso])


def merge_to_markdown(
    base_dir: Path,
    items: List[FileItem],
    merged_path: Path,
    max_bytes_per_file: int,
) -> None:
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    header = []
    header.append("# Merged Output for AI")
    header.append("")
    header.append(f"- base_dir: `{str(base_dir)}`")
    header.append(f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`")
    header.append(f"- include_ext: `{', '.join(sorted({it.ext for it in items if it.included}))}`")
    header.append(f"- max_bytes_per_file: `{max_bytes_per_file}`")
    header.append("")
    header.append("## File Index (included)")
    header.append("")
    for it in items:
        if it.included:
            header.append(f"- `{it.rel_path}` ({it.size_bytes} bytes, mtime={it.mtime_iso})")
    header.append("")
    header.append("## Contents")
    header.append("")

    with merged_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("\n".join(header) + "\n")

        for it in items:
            if not it.included:
                continue

            p = Path(it.abs_path)
            text, enc = try_read_text(p, max_bytes=max_bytes_per_file)

            out.write("\n\n---\n")
            out.write(f"### {it.rel_path}\n")
            out.write(f"- abs_path: `{it.abs_path}`\n")
            out.write(f"- size_bytes: `{it.size_bytes}`\n")
            out.write(f"- mtime: `{it.mtime_iso}`\n")
            out.write(f"- encoding: `{enc}`\n\n")

            if text is None:
                out.write("<!-- SKIPPED: could not read as text -->\n")
                continue

            fence = ""
            if it.ext in [".json", ".jsonl"]:
                fence = "json"
            elif it.ext == ".csv":
                fence = "csv"
            elif it.ext == ".html":
                fence = "html"
            elif it.ext in [".log", ".txt", ".md"]:
                fence = ""

            out.write(f"```{fence}\n")
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")
            out.write("```\n")


def make_zip(zip_path: Path, files: List[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    default_base = str(OUTPUT_DIR)
    default_out = str(Path(OUTPUT_DIR) / "merged_for_ai")
    ap.add_argument("--base", default=default_base, help="Base directory to scan")
    ap.add_argument("--out", default=default_out, help="Output directory")
    ap.add_argument("--include-ext", default=",".join(TEXT_EXT_DEFAULT),
                    help="Comma-separated extensions to include (e.g. .txt,.log,.json,.csv,.html)")
    ap.add_argument("--exclude-dir", default="",
                    help="Comma-separated directory names to exclude (e.g. .git,__pycache__)")
    ap.add_argument("--max-bytes-per-file", type=int, default=2_000_000,
                    help="Max bytes to read per file (default: 2,000,000)")
    ap.add_argument("--sort", choices=["path", "mtime"], default="path",
                    help="Sort order for merged output")
    ap.add_argument("--dry-run", action="store_true", help="Only create manifest (no merged file)")
    args = ap.parse_args()

    base_dir = Path(args.base)
    out_dir = Path(args.out)
    include_ext = [e.strip().lower() for e in args.include_ext.split(",") if e.strip()]
    exclude_dirs = {d.strip() for d in args.exclude_dir.split(",") if d.strip()}

    if not base_dir.exists():
        print(f"[ERROR] base_dir not found: {base_dir}", file=sys.stderr)
        return 2

    all_files = enumerate_files(base_dir)

    items: List[FileItem] = []
    for p in all_files:
        rel = normalize_rel(base_dir, p)
        if exclude_dirs:
            parts = Path(rel).parts
            if any(part in exclude_dirs for part in parts):
                items.append(FileItem(rel, str(p), p.suffix.lower(), p.stat().st_size, iso_mtime(p),
                                      included=False, reason="excluded_dir"))
                continue

        ext = p.suffix.lower()
        if ext in include_ext:
            items.append(FileItem(rel, str(p), ext, p.stat().st_size, iso_mtime(p),
                                  included=True, reason="ok"))
        else:
            items.append(FileItem(rel, str(p), ext, p.stat().st_size, iso_mtime(p),
                                  included=False, reason="ext_not_included"))

    if args.sort == "path":
        items.sort(key=lambda x: x.rel_path)
    else:
        items.sort(key=lambda x: x.mtime_iso)

    manifest_path = out_dir / "manifest.csv"
    merged_path = out_dir / "merged_for_ai.md"
    zip_path = out_dir / "merged_for_ai.zip"

    write_manifest(items, manifest_path)
    print(f"[OK] manifest: {manifest_path}")

    if args.dry_run:
        print("[INFO] dry-run enabled: skip merging")
        return 0

    merge_to_markdown(
        base_dir=base_dir,
        items=items,
        merged_path=merged_path,
        max_bytes_per_file=args.max_bytes_per_file,
    )
    print(f"[OK] merged: {merged_path}")

    make_zip(zip_path, [manifest_path, merged_path])
    print(f"[OK] zip: {zip_path}")

    inc = sum(1 for it in items if it.included)
    exc = len(items) - inc
    print(f"[SUMMARY] included={inc} excluded={exc} total={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
