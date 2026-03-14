from __future__ import annotations

import csv
import shutil
from pathlib import Path

from arena.artifacts.hash_utils import sha256_file
from arena.artifacts.discovery import iso_mtime, resolve_ai_source_path
from arena.artifacts.models import AICandidateStatus, AIManifestRecord
from arena.artifacts.policies import AI_FILES_SUBDIR, check_ai_export_exclusion, infer_category
from arena.artifacts.schema import validate_candidate_status_rows, validate_manifest_record_rows
from arena.artifacts.selection import candidate_map_by_relative_path, get_ai_candidate_files


def to_flat_export_filename(relative_path: str) -> str:
    return Path(relative_path).name


def resolve_flat_destination_path(export_dir: Path, relative_path: str) -> Path:
    flat_dir = export_dir / AI_FILES_SUBDIR
    flat_dir.mkdir(parents=True, exist_ok=True)

    filename = to_flat_export_filename(relative_path)
    candidate = flat_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for index in range(1, 1000):
        retry = flat_dir / f"{stem}__{index:03d}{suffix}"
        if not retry.exists():
            return retry

    raise RuntimeError(f"could not resolve destination filename for {relative_path}")


def build_ai_manifest_record(
    base_dir: Path,
    export_dir: Path,
    relative_path: str,
    required_level: str,
    copied_source_map: dict[str, str],
) -> AIManifestRecord:
    category = infer_category(relative_path)
    source_path, source_note = resolve_ai_source_path(base_dir, relative_path)

    if source_path is None:
        status = "missing_required" if required_level == "required" else "missing_recommended"
        return AIManifestRecord(
            relative_path=relative_path,
            category=category,
            required_level=required_level,
            exists=False,
            copied=False,
            size_bytes=0,
            mtime="",
            status=status,
            note=source_note,
        )

    source_real = str(source_path.resolve())
    duplicate_of = copied_source_map.get(source_real)
    size_bytes = source_path.stat().st_size
    mtime = iso_mtime(source_path)
    excluded, exclude_note = check_ai_export_exclusion(relative_path=relative_path, source_path=source_path)
    if excluded:
        note_parts = [exclude_note, f"source={source_real}"]
        if source_note:
            note_parts.insert(0, source_note)
        return AIManifestRecord(
            relative_path=relative_path,
            category=category,
            required_level=required_level,
            exists=True,
            copied=False,
            size_bytes=size_bytes,
            mtime=mtime,
            status="excluded_by_rule",
            note="; ".join(part for part in note_parts if part),
            source_path=source_real,
        )

    if duplicate_of is not None:
        return AIManifestRecord(
            relative_path=relative_path,
            category=category,
            required_level=required_level,
            exists=True,
            copied=False,
            size_bytes=size_bytes,
            mtime=mtime,
            status="duplicate_source_skipped",
            note=f"duplicate_source_of={duplicate_of}; source={source_real}",
            source_path=source_real,
        )

    destination = resolve_flat_destination_path(export_dir, relative_path)
    try:
        shutil.copy2(source_path, destination)
    except Exception as exc:
        return AIManifestRecord(
            relative_path=relative_path,
            category=category,
            required_level=required_level,
            exists=True,
            copied=False,
            size_bytes=size_bytes,
            mtime=mtime,
            status="copy_failed",
            note=str(exc),
            source_path=source_real,
        )

    copied_source_map[source_real] = relative_path
    copied_rel = destination.relative_to(export_dir).as_posix()
    artifact_sha256 = sha256_file(destination)
    return AIManifestRecord(
        relative_path=relative_path,
        category=category,
        required_level=required_level,
        exists=True,
        copied=True,
        size_bytes=size_bytes,
        mtime=mtime,
        status="ok",
        note=", ".join(part for part in [source_note, f"copied_as={copied_rel}", f"source={source_real}"] if part),
        source_path=source_real,
        copied_path=copied_rel,
        artifact_sha256=artifact_sha256,
    )


def write_ai_selected_manifest(records: list[AIManifestRecord], manifest_path: Path) -> None:
    rows = [
        {
            "relative_path": record.relative_path,
            "category": record.category,
            "required_level": record.required_level,
            "exists": int(record.exists),
            "copied": int(record.copied),
            "size_bytes": record.size_bytes,
            "mtime": record.mtime,
            "status": record.status,
            "note": record.note,
            "artifact_sha256": record.artifact_sha256,
        }
        for record in records
    ]
    validate_manifest_record_rows(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "relative_path",
                "category",
                "required_level",
                "exists",
                "copied",
                "size_bytes",
                "mtime",
                "status",
                "note",
                "artifact_sha256",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["relative_path"],
                    row["category"],
                    row["required_level"],
                    row["exists"],
                    row["copied"],
                    row["size_bytes"],
                    row["mtime"],
                    row["status"],
                    row["note"],
                    row["artifact_sha256"],
                ]
            )


def collect_candidate_status(base_dir: Path, records: list[AIManifestRecord]) -> list[AICandidateStatus]:
    copied_by_relative = {record.relative_path: record for record in records if record.copied}
    copied_source_paths = {record.source_path for record in records if record.copied and record.source_path}
    records_by_relative = {record.relative_path: record for record in records}
    result: list[AICandidateStatus] = []

    for candidate in get_ai_candidate_files():
        source_path, source_note = resolve_ai_source_path(base_dir, candidate.relative_path)
        exists = source_path is not None
        copied_to_export = False
        resolved_source = ""
        note_parts: list[str] = []

        if exists and source_path is not None:
            resolved_source = str(source_path.resolve())
            copied_to_export = candidate.relative_path in copied_by_relative or resolved_source in copied_source_paths
            note_parts.append("resolved")
            if source_note:
                note_parts.append(source_note)
            record = records_by_relative.get(candidate.relative_path)
            if record is not None:
                note_parts.append(f"manifest_status={record.status}")
            if not copied_to_export:
                note_parts.append("candidate_only_or_not_copied")
        else:
            note_parts.append("missing candidate")
            if source_note:
                note_parts.append(source_note)

        result.append(
            AICandidateStatus(
                logical_name=candidate.logical_name,
                relative_path=candidate.relative_path,
                expected_path=candidate.expected_path,
                exists=exists,
                copied_to_export=copied_to_export,
                priority=candidate.priority,
                note="; ".join(note_parts),
                source_path=resolved_source,
                reason=candidate.reason,
                what_it_enables=candidate.what_it_enables,
                rationale=candidate.rationale,
            )
        )
    return result


def write_candidate_status_csv(statuses: list[AICandidateStatus], csv_path: Path) -> None:
    rows = [
        {
            "logical_name": status.logical_name,
            "expected_path": status.expected_path,
            "exists": int(status.exists),
            "copied_to_export": int(status.copied_to_export),
            "priority": status.priority,
            "note": status.note,
        }
        for status in statuses
    ]
    validate_candidate_status_rows(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["logical_name", "expected_path", "exists", "copied_to_export", "priority", "note"])
        for row in rows:
            writer.writerow(
                [
                    row["logical_name"],
                    row["expected_path"],
                    row["exists"],
                    row["copied_to_export"],
                    row["priority"],
                    row["note"],
                ]
            )


def write_ai_selected_manifest_extended(
    records: list[AIManifestRecord],
    candidate_statuses: list[AICandidateStatus],
    manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_by_rel = candidate_map_by_relative_path()
    seen_relative_paths: set[str] = set()
    rows: list[dict[str, object]] = []

    for record in records:
        candidate = candidate_by_rel.get(record.relative_path)
        rows.append(
            {
                "relative_path": record.relative_path,
                "category": record.category,
                "required_level": record.required_level,
                "exists": int(record.exists),
                "copied": int(record.copied),
                "size_bytes": record.size_bytes,
                "mtime": record.mtime,
                "status": record.status,
                "note": record.note,
                "artifact_sha256": record.artifact_sha256,
                "priority": candidate.priority if candidate is not None else ("A" if record.required_level == "required" else "B"),
                "rationale": candidate.rationale if candidate is not None else "",
                "candidate_only": 0,
                "logical_name": candidate.logical_name if candidate is not None else "",
                "expected_path": candidate.expected_path if candidate is not None else "",
            }
        )
        seen_relative_paths.add(record.relative_path)

    for status in candidate_statuses:
        if status.relative_path in seen_relative_paths:
            continue
        rows.append(
            {
                "relative_path": status.relative_path,
                "category": "statistics_candidate",
                "required_level": "candidate",
                "exists": int(status.exists),
                "copied": int(status.copied_to_export),
                "size_bytes": 0,
                "mtime": "",
                "status": "candidate_only_found" if status.exists else "missing_candidate",
                "note": status.note,
                "artifact_sha256": "",
                "priority": status.priority,
                "rationale": status.rationale,
                "candidate_only": 1,
                "logical_name": status.logical_name,
                "expected_path": status.expected_path,
            }
        )

    validate_manifest_record_rows(rows)

    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "relative_path",
                "category",
                "required_level",
                "exists",
                "copied",
                "size_bytes",
                "mtime",
                "status",
                "note",
                "artifact_sha256",
                "priority",
                "rationale",
                "candidate_only",
                "logical_name",
                "expected_path",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["relative_path"],
                    row["category"],
                    row["required_level"],
                    row["exists"],
                    row["copied"],
                    row["size_bytes"],
                    row["mtime"],
                    row["status"],
                    row["note"],
                    row["artifact_sha256"],
                    row["priority"],
                    row["rationale"],
                    row["candidate_only"],
                    row["logical_name"],
                    row["expected_path"],
                ]
            )

