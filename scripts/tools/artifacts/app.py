from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from arena.lib.runtime_config import load_settings

from arena.artifacts.discovery import enumerate_files, normalize_rel, resolve_optional_export_source
from scripts.tools.artifacts.documentation import (
    write_ai_change_point_note,
    write_ai_export_summary,
    write_ai_settings_snapshot,
    write_analysis_design_note,
    write_analysis_methodology,
    write_hardware_date_recommendation,
    write_needed_files_for_statistics,
)
from arena.artifacts.hash_utils import compute_bundle_sha256, read_artifact_hashes, update_record_hashes, write_artifact_hashes
from arena.artifacts.integrity import run_ai_export_integrity_check, write_integrity_summary_json
from arena.artifacts.lineage import write_artifact_lineage
from arena.artifacts.manifest import (
    build_ai_manifest_record,
    collect_candidate_status,
    write_ai_selected_manifest,
    write_ai_selected_manifest_extended,
    write_candidate_status_csv,
)
from arena.artifacts.models import AIManifestRecord, FileItem
from scripts.tools.artifacts.packaging import create_ai_review_packs, resolve_timestamped_export_dir
from arena.artifacts.policies import (
    AI_ARTIFACT_INDEX_FILENAME,
    AI_ARTIFACT_LINEAGE_FILENAME,
    AI_ARTIFACT_PROVENANCE_FILENAME,
    AI_CANDIDATE_STATUS_CSV_FILENAME,
    AI_EXPORT_DIR_PREFIX,
    AI_ARTIFACT_HASHES_FILENAME,
    AI_INTEGRITY_SUMMARY_JSON_FILENAME,
    AI_MANIFEST_EXTENDED_FILENAME,
    AI_MANIFEST_FILENAME,
    AI_PACK_DIR_GEMINI,
    AI_PACK_DIR_GPT,
    AI_PACK_DIR_GROK,
    AI_RUN_METADATA_FILENAME,
    AI_SUMMARY_FILENAME,
    ARTIFACT_SUBSYSTEM_VERSION,
    ALWAYS_EXCLUDE_DIRS,
    ALWAYS_EXCLUDE_DIR_PREFIXES,
    ALWAYS_EXCLUDE_REL_PATHS,
    NORMAL_MODE_OPTIONAL_EXPORT_FILES,
)
from arena.artifacts.provenance import write_artifact_provenance
from arena.artifacts.repro_stamp import resolve_generated_at, write_reproducibility_stamp
from arena.artifacts.run_metadata import write_run_metadata
from arena.artifacts.schema import export_schema_catalog, validate_artifact_index
from arena.artifacts.selection import get_ai_candidate_files, iter_ai_targets


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def try_read_text(path: Path, max_bytes: int, prefer_tail: bool = False) -> tuple[str | None, str]:
    size = path.stat().st_size
    to_read = min(size, max_bytes)

    with path.open("rb") as file:
        head = file.read(min(to_read, 4096))
        if b"\x00" in head:
            return None, "binary_like(NULL found)"

    if prefer_tail and size > max_bytes:
        with path.open("rb") as file:
            file.seek(size - to_read)
            raw = file.read(to_read)
        if b"\n" in raw:
            raw = raw.split(b"\n", 1)[1]
    else:
        with path.open("rb") as file:
            raw = file.read(to_read)

    encodings = ["utf-8-sig", "utf-8", "cp932", "latin-1"]
    text: str | None = None
    note: str | None = None
    last_err: Exception | None = None
    for encoding in encodings:
        try:
            text = raw.decode(encoding, errors="strict")
            note = encoding
            break
        except Exception as exc:
            last_err = exc

    if text is None and prefer_tail:
        for encoding in encodings:
            try:
                text = raw.decode(encoding, errors="replace")
                note = f"{encoding}(replace)"
                break
            except Exception:
                continue

    if text is None:
        return None, f"decode_failed({last_err})"

    if size > max_bytes:
        if prefer_tail:
            text += f"\n\n<!-- TRUNCATED: showing tail, file_size={size} bytes exceeded max_bytes={max_bytes} -->\n"
        else:
            text += f"\n\n<!-- TRUNCATED: file_size={size} bytes exceeded max_bytes={max_bytes} -->\n"

    return text, note or "unknown"


def write_manifest(items: list[FileItem], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["included", "reason", "rel_path", "abs_path", "ext", "size_bytes", "mtime_iso"])
        for item in items:
            writer.writerow([int(item.included), item.reason, item.rel_path, item.abs_path, item.ext, item.size_bytes, item.mtime_iso])


def merge_to_markdown(
    base_dir: Path,
    items: list[FileItem],
    merged_path: Path,
    max_bytes_per_file: int,
    deterministic: bool = False,
) -> None:
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# Merged Output for AI",
        "",
        f"- base_dir: `{base_dir}`",
        f"- generated_at: `{resolve_generated_at(deterministic)}`",
        f"- include_ext: `{', '.join(sorted({item.ext for item in items if item.included}))}`",
        f"- max_bytes_per_file: `{max_bytes_per_file}`",
        "",
        "## File Index (included)",
        "",
    ]
    for item in items:
        if item.included:
            header.append(f"- `{item.rel_path}` ({item.size_bytes} bytes, mtime={item.mtime_iso})")
    header.extend(["", "## Contents", ""])

    with merged_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("\n".join(header) + "\n")
        for item in items:
            if not item.included:
                continue
            path = Path(item.abs_path)
            prefer_tail = item.ext in {".jsonl", ".log"}
            text, encoding = try_read_text(path, max_bytes=max_bytes_per_file, prefer_tail=prefer_tail)

            out.write("\n\n---\n")
            out.write(f"### {item.rel_path}\n")
            out.write(f"- abs_path: `{item.abs_path}`\n")
            out.write(f"- size_bytes: `{item.size_bytes}`\n")
            out.write(f"- mtime: `{item.mtime_iso}`\n")
            out.write(f"- encoding: `{encoding}`\n\n")

            if text is None:
                out.write("<!-- SKIPPED: could not read as text -->\n")
                continue

            fence = ""
            if item.ext in {".json", ".jsonl"}:
                fence = "json"
            elif item.ext == ".csv":
                fence = "csv"
            elif item.ext == ".html":
                fence = "html"

            out.write(f"```{fence}\n")
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")
            out.write("```\n")


def make_zip(zip_path: Path, files: list[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, arcname=file.name)


def copy_optional_export_files(base_dir: Path, out_dir: Path, relative_paths: list[str]) -> list[Path]:
    copied: list[Path] = []
    for rel in relative_paths:
        src = resolve_optional_export_source(base_dir, rel)
        if src is None:
            print(f"[WARN] optional file not found: {rel}")
            continue
        dst = out_dir / Path(rel).name
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)
            print(f"[OK] optional file: {dst} (source={src})")
        except Exception as exc:
            print(f"[WARN] optional file copy failed: {rel} ({exc})")
    return copied


def write_artifact_index(
    export_dir: Path,
    generated_at: str,
    records: list[AIManifestRecord],
    bundle_sha256: str,
) -> Path:
    payload = {
        "bundle_id": bundle_sha256,
        "timestamp": generated_at,
        "artifact_count": sum(1 for record in records if record.copied),
        "bundle_sha256": bundle_sha256,
        "run_metadata_ref": AI_RUN_METADATA_FILENAME,
        "provenance_ref": AI_ARTIFACT_PROVENANCE_FILENAME,
        "integrity_ref": AI_INTEGRITY_SUMMARY_JSON_FILENAME,
        "lineage_ref": AI_ARTIFACT_LINEAGE_FILENAME,
        "manifest_ref": AI_MANIFEST_FILENAME,
        "schema_version": ARTIFACT_SUBSYSTEM_VERSION,
    }
    validate_artifact_index(payload)
    destination = export_dir / AI_ARTIFACT_INDEX_FILENAME
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return destination


def export_ai_folder(
    base_dir: Path,
    output_root: Path,
    deterministic: bool = False,
) -> tuple[Path, list[AIManifestRecord]]:
    export_dir = resolve_timestamped_export_dir(output_root, AI_EXPORT_DIR_PREFIX)
    copied_source_map: dict[str, str] = {}
    records = [
        build_ai_manifest_record(base_dir, export_dir, relative_path, required_level, copied_source_map)
        for relative_path, required_level in iter_ai_targets(base_dir)
    ]
    level_order = {"required": 0, "recommended": 1}
    records.sort(key=lambda record: (level_order.get(record.required_level, 99), record.relative_path))

    manifest_path = export_dir / AI_MANIFEST_FILENAME
    extended_manifest_path = export_dir / AI_MANIFEST_EXTENDED_FILENAME
    summary_path = export_dir / AI_SUMMARY_FILENAME
    candidate_status_csv_path = export_dir / AI_CANDIDATE_STATUS_CSV_FILENAME
    generated_at = resolve_generated_at(deterministic)
    settings_json_path, settings_md_path = write_ai_settings_snapshot(export_dir, generated_at)
    settings = load_settings(force_reload=False)
    change_point_note_path = write_ai_change_point_note(export_dir, settings.path)
    hardware_note_path = write_hardware_date_recommendation(export_dir)
    needed_files_note_path = write_needed_files_for_statistics(export_dir, get_ai_candidate_files())
    analysis_design_path = write_analysis_design_note(export_dir, hardware_note_path, change_point_note_path)
    analysis_methodology_path = write_analysis_methodology(export_dir)
    candidate_statuses = collect_candidate_status(base_dir, records)
    write_candidate_status_csv(candidate_statuses, candidate_status_csv_path)
    update_record_hashes(export_dir, records)
    artifact_hashes_path = write_artifact_hashes(export_dir / AI_ARTIFACT_HASHES_FILENAME, records)
    repro_stamp_path = write_reproducibility_stamp(export_dir, export_mode="ai_export", deterministic=deterministic)

    write_ai_selected_manifest(records, manifest_path)
    write_ai_selected_manifest_extended(records, candidate_statuses, extended_manifest_path)
    provenance_path = write_artifact_provenance(export_dir, records, generated_at)
    run_metadata_path = write_run_metadata(export_dir, records, generated_at, deterministic)
    lineage_path = write_artifact_lineage(export_dir, records, generated_at)
    integrity = run_ai_export_integrity_check(records, export_dir=export_dir, hash_manifest_path=artifact_hashes_path)
    integrity_summary_path = write_integrity_summary_json(export_dir, integrity)
    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        manifest_rows = list(csv.DictReader(manifest_file))
    bundle_sha256 = compute_bundle_sha256(
        artifact_hashes=read_artifact_hashes(artifact_hashes_path),
        manifest_rows=manifest_rows,
        schema_payload=export_schema_catalog(),
    )
    artifact_index_path = write_artifact_index(export_dir, generated_at, records, bundle_sha256)
    write_ai_export_summary(
        summary_path=summary_path,
        generated_at=generated_at,
        source_base_dir=base_dir,
        export_dir=export_dir,
        records=records,
        settings_json_path=settings_json_path,
        settings_md_path=settings_md_path,
        change_point_note_path=change_point_note_path,
        integrity=integrity,
        hardware_note_path=hardware_note_path,
        needed_files_note_path=needed_files_note_path,
        candidate_status_csv_path=candidate_status_csv_path,
        extended_manifest_path=extended_manifest_path,
        candidate_statuses=candidate_statuses,
        provenance_path=provenance_path,
        run_metadata_path=run_metadata_path,
        lineage_path=lineage_path,
        integrity_summary_path=integrity_summary_path,
        artifact_index_path=artifact_index_path,
        bundle_sha256=bundle_sha256,
        deterministic=deterministic,
    )
    create_ai_review_packs(
        export_dir=export_dir,
        records=records,
        generated_paths=[
            manifest_path,
            extended_manifest_path,
            summary_path,
            artifact_hashes_path,
            repro_stamp_path,
            provenance_path,
            run_metadata_path,
            lineage_path,
            integrity_summary_path,
            artifact_index_path,
            settings_json_path,
            settings_md_path,
            change_point_note_path,
            hardware_note_path,
            needed_files_note_path,
            analysis_design_path,
            analysis_methodology_path,
            candidate_status_csv_path,
        ],
        deterministic=deterministic,
    )
    return export_dir, records


def resolve_ai_export_root(out_dir: Path, ai_export_root: str, use_out_parent: bool) -> Path:
    if ai_export_root:
        return Path(ai_export_root)
    return out_dir


def run_ai_export_with_summary(
    base_dir: Path,
    output_root: Path,
    deterministic: bool = False,
) -> tuple[Path, list[AIManifestRecord]]:
    export_dir, records = export_ai_folder(base_dir, output_root, deterministic=deterministic)
    manifest_path = export_dir / AI_MANIFEST_FILENAME
    extended_manifest_path = export_dir / AI_MANIFEST_EXTENDED_FILENAME
    summary_path = export_dir / AI_SUMMARY_FILENAME
    candidate_status_csv_path = export_dir / AI_CANDIDATE_STATUS_CSV_FILENAME
    gemini_pack_dir = export_dir / AI_PACK_DIR_GEMINI
    gpt_pack_dir = export_dir / AI_PACK_DIR_GPT
    grok_pack_dir = export_dir / AI_PACK_DIR_GROK
    required_missing = sum(1 for record in records if record.status == "missing_required")
    recommended_missing = sum(1 for record in records if record.status == "missing_recommended")
    duplicate_skipped = sum(1 for record in records if record.status == "duplicate_source_skipped")
    copied_total = sum(1 for record in records if record.copied)
    print(f"[OK] ai_export_dir: {export_dir}")
    print(f"[OK] ai_manifest: {manifest_path}")
    print(f"[OK] ai_manifest_extended: {extended_manifest_path}")
    print(f"[OK] ai_summary: {summary_path}")
    print(f"[OK] ai_candidate_status: {candidate_status_csv_path}")
    print(f"[OK] ai_pack_gemini: {gemini_pack_dir}")
    print(f"[OK] ai_pack_gpt: {gpt_pack_dir}")
    print(f"[OK] ai_pack_grok: {grok_pack_dir}")
    print(
        f"[SUMMARY] copied_total={copied_total} "
        f"required_missing={required_missing} recommended_missing={recommended_missing} "
        f"duplicate_source_skipped={duplicate_skipped}"
    )
    return export_dir, records


def run_from_args(args: argparse.Namespace) -> int:
    base_dir = Path(args.base)
    out_dir = Path(args.out)
    include_ext = [ext.strip().lower() for ext in args.include_ext.split(",") if ext.strip()]
    exclude_dirs = {directory.strip() for directory in args.exclude_dir.split(",") if directory.strip()}
    exclude_dirs.update(ALWAYS_EXCLUDE_DIRS)

    if not base_dir.exists():
        print(f"[ERROR] base_dir not found: {base_dir}", file=sys.stderr)
        return 2

    if args.export_ai_folder:
        if args.ai_export_use_out_parent and not args.ai_export_root:
            print("[WARN] --ai-export-use-out-parent is deprecated. Output still uses --out.")
        output_root = resolve_ai_export_root(out_dir, args.ai_export_root, args.ai_export_use_out_parent)
        run_ai_export_with_summary(base_dir, output_root, deterministic=args.deterministic)
        return 0

    items: list[FileItem] = []
    for path in enumerate_files(base_dir):
        rel = normalize_rel(base_dir, path)
        if rel in ALWAYS_EXCLUDE_REL_PATHS:
            items.append(FileItem(rel, str(path), path.suffix.lower(), path.stat().st_size, iso_mtime(path), False, "excluded_file"))
            continue

        if exclude_dirs:
            parts = Path(rel).parts
            if any(part in exclude_dirs or part.startswith(ALWAYS_EXCLUDE_DIR_PREFIXES) for part in parts):
                items.append(FileItem(rel, str(path), path.suffix.lower(), path.stat().st_size, iso_mtime(path), False, "excluded_dir"))
                continue

        ext = path.suffix.lower()
        included = ext in include_ext
        items.append(
            FileItem(
                rel_path=rel,
                abs_path=str(path),
                ext=ext,
                size_bytes=path.stat().st_size,
                mtime_iso=iso_mtime(path),
                included=included,
                reason="ok" if included else "ext_not_included",
            )
        )

    if args.sort == "path":
        items.sort(key=lambda item: item.rel_path)
    else:
        if args.deterministic:
            items.sort(key=lambda item: (item.mtime_iso, item.rel_path))
        else:
            items.sort(key=lambda item: item.mtime_iso)

    manifest_path = out_dir / "manifest.csv"
    merged_path = out_dir / "merged_for_ai.md"
    zip_path = out_dir / "merged_for_ai.zip"

    write_manifest(items, manifest_path)
    print(f"[OK] manifest: {manifest_path}")

    if args.dry_run:
        print("[INFO] dry-run enabled: skipping merged output generation")
        return 0

    merge_to_markdown(base_dir, items, merged_path, args.max_bytes_per_file, deterministic=args.deterministic)
    print(f"[OK] merged: {merged_path}")

    optional_files = copy_optional_export_files(base_dir, out_dir, NORMAL_MODE_OPTIONAL_EXPORT_FILES)
    make_zip(zip_path, [manifest_path, merged_path, *optional_files])
    print(f"[OK] zip: {zip_path}")

    included_count = sum(1 for item in items if item.included)
    excluded_count = len(items) - included_count
    print(f"[SUMMARY] included={included_count} excluded={excluded_count} total={len(items)}")

    if args.no_ai_export:
        print("[INFO] --no-ai-export specified: skipping AI export folder generation")
        return 0

    if args.ai_export_use_out_parent and not args.ai_export_root:
        print("[WARN] --ai-export-use-out-parent is deprecated. Output still uses --out.")
    output_root = resolve_ai_export_root(out_dir, args.ai_export_root, args.ai_export_use_out_parent)
    run_ai_export_with_summary(base_dir, output_root, deterministic=args.deterministic)
    return 0
