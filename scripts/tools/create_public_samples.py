#!/usr/bin/env python3
"""Create public sample datasets and optionally delete original files.

Behavior:
- Scan `data/` and `output/` recursively.
- For JSONL/TXT: keep first 200 lines.
- For CSV: keep first 200 rows.
- Save sampled files to `<root>/sample/` with `sample_` prefix.
- Skip files already under `sample/`.
- Mark every non-sample file for deletion (including unsupported types).
- Ask for confirmation before deleting files.
- Never delete directories.
"""

import csv
import os
from itertools import islice
from pathlib import Path


MAX_ITEMS = 200
TARGET_DIRS = ("data", "output")
SUPPORTED_SUFFIXES = {".csv", ".txt", ".jsonl"}


def is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def is_in_sample_dir(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return bool(rel.parts) and rel.parts[0].lower() == "sample"


def iter_all_files(root: Path):
    if not root.exists():
        return []
    return [p for p in sorted(root.rglob("*")) if p.is_file()]


def sample_name_from_relative(root: Path, src: Path) -> str:
    rel = src.relative_to(root)
    flattened = "_".join(rel.parts)
    return f"sample_{flattened}"


def sample_text_like(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8", errors="replace") as rf:
        with dst.open("w", encoding="utf-8", newline="") as wf:
            for line in islice(rf, MAX_ITEMS):
                wf.write(line)


def sample_csv(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8", errors="replace", newline="") as rf:
        reader = csv.reader(rf)
        with dst.open("w", encoding="utf-8", newline="") as wf:
            writer = csv.writer(wf)
            for row in islice(reader, MAX_ITEMS):
                writer.writerow(row)


def create_sample(src: Path, dst: Path) -> None:
    if src.suffix.lower() == ".csv":
        sample_csv(src, dst)
        return
    sample_text_like(src, dst)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    sampled = []
    skipped = []
    to_delete = []

    for dir_name in TARGET_DIRS:
        root = repo_root / dir_name
        if not root.exists():
            print(f"[SKIPPED] {dir_name}/ (directory not found)")
            continue

        sample_dir = root / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for src in iter_all_files(root):
            rel = src.relative_to(repo_root)

            if is_in_sample_dir(src, root):
                skipped.append((src, "already in sample/"))
                print(f"[SKIPPED] {rel} (already in sample/)")
                continue

            if is_supported_file(src):
                dst = sample_dir / sample_name_from_relative(root, src)
                create_sample(src, dst)
                sampled.append((src, dst))
                print(f"[SAMPLED] {rel} -> {dst.relative_to(repo_root)}")
            else:
                skipped.append((src, "unsupported for sampling; delete only"))
                print(f"[SKIPPED] {rel} (unsupported for sampling; delete only)")

            to_delete.append(src)

    if not to_delete:
        print("Nothing to delete. Repository already looks sample-only under data/output.")
        print(f"Summary: sampled={len(sampled)} deleted=0 skipped={len(skipped)}")
        return

    print("")
    print("Files marked for deletion:")
    for src in to_delete:
        print(f"  - {src.relative_to(repo_root)}")

    answer = input("Proceed with deleting original files? [y/N] ").strip().lower()
    if answer != "y":
        print("Deletion canceled. Sample files were kept.")
        print(f"Summary: sampled={len(sampled)} deleted=0 skipped={len(skipped)}")
        return

    deleted = 0
    for src in to_delete:
        try:
            src.unlink()
            deleted += 1
            print(f"[DELETED] {src.relative_to(repo_root)}")
        except OSError as exc:
            print(f"[SKIPPED] {src.relative_to(repo_root)} (delete failed: {exc})")

    print("")
    print(f"Summary: sampled={len(sampled)} deleted={deleted} skipped={len(skipped)}")


if __name__ == "__main__":
    main()
