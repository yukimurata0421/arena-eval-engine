from __future__ import annotations

import sqlite3
from pathlib import Path

from arena.synthesis.ingest_artifacts import (
    build_artifact_path,
    log_repair_event,
    sha256_file,
    sha256_text,
    write_bytes_once,
    write_text_once,
)
from arena.synthesis.ingest_models import (
    MODEL_CHOICES,
    IngestFileResult,
    IngestRawReport,
    RepairIngestError,
    RepairValidationError,
)
from arena.synthesis.ingest_store import (
    connect_for_ingest,
    delete_existing_source_claims,
    ensure_tables,
    ingest_records,
    upsert_ingested_file,
)
from arena.synthesis.ingest_validation import (
    downgrade_semantic_errors_to_warnings,
    extract_source_date,
    infer_model_from_path,
    load_and_validate,
    render_validation_summary,
    repair_records,
)
from arena.synthesis.paths import RAW_DIR, RAW_ORIGINAL_DIR, RAW_REPAIRED_DIR, REPAIR_LOG_DIR
from arena.synthesis.validator import validate_records


def _ingest_file_impl(
    path: Path,
    model: str | None = None,
    raw_root: Path = RAW_DIR,
    db_path: Path | None = None,
    repair: bool = False,
    repair_semantic: bool = False,
    raw_original_dir: Path = RAW_ORIGINAL_DIR,
    raw_repaired_dir: Path = RAW_REPAIRED_DIR,
    repair_log_dir: Path = REPAIR_LOG_DIR,
) -> IngestFileResult:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Ingest source file does not exist or is not a file: {path}")
    if repair_semantic and not repair:
        raise ValueError("--repair-semantic requires --repair")

    source_path = path.resolve()
    raw_root_resolved = raw_root.resolve()
    source_hash = sha256_file(source_path)
    source_date = extract_source_date(source_path)
    if source_date is None:
        raise ValueError(
            "invalid source filename: "
            f"{source_path.name} (expected YYYYMMDD.json/.jsonl or YYYYMMDD-N.json/.jsonl, N>=1)"
        )
    resolved_model = model or infer_model_from_path(source_path, raw_root=raw_root_resolved)

    source_bytes = source_path.read_bytes()
    source_text = source_bytes.decode("utf-8-sig")

    original_path: Path | None = None
    original_saved = 0
    repaired_path: Path | None = None
    repaired_saved = 0

    if repair:
        original_path = build_artifact_path(
            raw_original_dir,
            source_path=source_path,
            raw_root=raw_root_resolved,
            source_hash=source_hash,
        )
        original_saved = write_bytes_once(original_path, source_bytes)

    with connect_for_ingest(db_path) as conn:
        ensure_tables(conn)
        existing = conn.execute(
            "SELECT source_sha256 FROM ingested_files WHERE source_path = ?",
            (str(source_path),),
        ).fetchone()
        is_duplicate = existing is not None and existing["source_sha256"] == source_hash

        if is_duplicate and not repair:
            return IngestFileResult(
                source_path=source_path,
                model=resolved_model,
                source_date=source_date,
                records_ingested=0,
                skipped_duplicate=True,
                warning_count=0,
            )

        if repair:
            repair_outcome = repair_records(
                source_path=source_path,
                source_text=source_text,
                repair_semantic=repair_semantic,
            )
            repaired_path = build_artifact_path(
                raw_repaired_dir,
                source_path=source_path,
                raw_root=raw_root_resolved,
                source_hash=source_hash,
            )
            repaired_saved = write_text_once(repaired_path, repair_outcome.repaired_text)
            repaired_sha256 = sha256_text(repair_outcome.repaired_text)

            if is_duplicate:
                log_repair_event(
                    repair_log_dir,
                    source_path=source_path,
                    source_hash=source_hash,
                    repaired_sha256=repaired_sha256,
                    original_path=original_path,
                    repaired_path=repaired_path,
                    repair_requested=True,
                    repair_semantic_requested=repair_semantic,
                    syntax_repairs_applied=repair_outcome.syntax_repairs,
                    shape_repairs_applied=repair_outcome.shape_repairs,
                    semantic_repairs_applied=repair_outcome.semantic_repairs,
                    warnings=repair_outcome.warnings,
                    validation_result="skipped_duplicate",
                    db_ingest_result="skipped_duplicate",
                    inserted_records=0,
                    failed_records=0,
                    record_count_before=repair_outcome.record_count_before,
                    record_count_after=repair_outcome.record_count_after,
                    skipped_reason="duplicate_source_sha256",
                )
                total_repairs = (
                    len(repair_outcome.syntax_repairs)
                    + len(repair_outcome.shape_repairs)
                    + len(repair_outcome.semantic_repairs)
                )
                return IngestFileResult(
                    source_path=source_path,
                    model=resolved_model,
                    source_date=source_date,
                    records_ingested=0,
                    skipped_duplicate=True,
                    warning_count=len(repair_outcome.warnings),
                    notes=tuple(repair_outcome.warnings),
                    original_saved=original_saved,
                    repaired_saved=repaired_saved,
                    repairs_applied=total_repairs,
                )

            if repair_outcome.error_message:
                log_repair_event(
                    repair_log_dir,
                    source_path=source_path,
                    source_hash=source_hash,
                    repaired_sha256=repaired_sha256,
                    original_path=original_path,
                    repaired_path=repaired_path,
                    repair_requested=True,
                    repair_semantic_requested=repair_semantic,
                    syntax_repairs_applied=repair_outcome.syntax_repairs,
                    shape_repairs_applied=repair_outcome.shape_repairs,
                    semantic_repairs_applied=repair_outcome.semantic_repairs,
                    warnings=[repair_outcome.error_message, *repair_outcome.warnings],
                    validation_result="not_run",
                    db_ingest_result="repair_failed",
                    inserted_records=0,
                    failed_records=1,
                    record_count_before=repair_outcome.record_count_before,
                    record_count_after=repair_outcome.record_count_after,
                )
                raise RepairIngestError(f"repair failed for {source_path}: {repair_outcome.error_message}")

            validation_issues = validate_records(repair_outcome.records)
            warning_messages = list(repair_outcome.warnings)
            if not repair_semantic:
                validation_issues, semantic_warnings = downgrade_semantic_errors_to_warnings(
                    records=repair_outcome.records,
                    issues=validation_issues,
                )
                warning_messages.extend(semantic_warnings)
            warning_messages.extend(issue.format() for issue in validation_issues if issue.level == "WARNING")
            validation_errors = [issue for issue in validation_issues if issue.level == "ERROR"]
            if validation_errors:
                log_repair_event(
                    repair_log_dir,
                    source_path=source_path,
                    source_hash=source_hash,
                    repaired_sha256=repaired_sha256,
                    original_path=original_path,
                    repaired_path=repaired_path,
                    repair_requested=True,
                    repair_semantic_requested=repair_semantic,
                    syntax_repairs_applied=repair_outcome.syntax_repairs,
                    shape_repairs_applied=repair_outcome.shape_repairs,
                    semantic_repairs_applied=repair_outcome.semantic_repairs,
                    warnings=warning_messages,
                    validation_result="failed",
                    db_ingest_result="not_ingested",
                    inserted_records=0,
                    failed_records=repair_outcome.record_count_after,
                    record_count_before=repair_outcome.record_count_before,
                    record_count_after=repair_outcome.record_count_after,
                )
                raise RepairValidationError(render_validation_summary(source_path, validation_issues))

            if existing is not None:
                delete_existing_source_claims(conn, source_path)

            ingested_count = ingest_records(
                conn,
                repair_outcome.records,
                source_path=source_path,
                model=resolved_model,
                source_date=source_date,
            )
            upsert_ingested_file(
                conn,
                source_path=source_path,
                source_hash=source_hash,
                model=resolved_model,
                source_date=source_date,
                record_count=ingested_count,
            )
            conn.commit()

            log_repair_event(
                repair_log_dir,
                source_path=source_path,
                source_hash=source_hash,
                repaired_sha256=repaired_sha256,
                original_path=original_path,
                repaired_path=repaired_path,
                repair_requested=True,
                repair_semantic_requested=repair_semantic,
                syntax_repairs_applied=repair_outcome.syntax_repairs,
                shape_repairs_applied=repair_outcome.shape_repairs,
                semantic_repairs_applied=repair_outcome.semantic_repairs,
                warnings=warning_messages,
                validation_result="passed",
                db_ingest_result="ingested",
                inserted_records=ingested_count,
                failed_records=0,
                record_count_before=repair_outcome.record_count_before,
                record_count_after=repair_outcome.record_count_after,
            )

            total_repairs = (
                len(repair_outcome.syntax_repairs)
                + len(repair_outcome.shape_repairs)
                + len(repair_outcome.semantic_repairs)
            )
            notes = tuple(warning_messages)
            if total_repairs > 0:
                notes = (*notes, f"repair applied: {source_path.name}")

            return IngestFileResult(
                source_path=source_path,
                model=resolved_model,
                source_date=source_date,
                records_ingested=ingested_count,
                skipped_duplicate=False,
                warning_count=len(warning_messages),
                notes=notes,
                original_saved=original_saved,
                repaired_saved=repaired_saved,
                repairs_applied=total_repairs,
            )

        records, warning_count = load_and_validate(source_path)
        if existing is not None:
            delete_existing_source_claims(conn, source_path)

        ingested_count = ingest_records(
            conn,
            records,
            source_path=source_path,
            model=resolved_model,
            source_date=source_date,
        )
        upsert_ingested_file(
            conn,
            source_path=source_path,
            source_hash=source_hash,
            model=resolved_model,
            source_date=source_date,
            record_count=ingested_count,
        )
        conn.commit()

    return IngestFileResult(
        source_path=source_path,
        model=resolved_model,
        source_date=source_date,
        records_ingested=ingested_count,
        skipped_duplicate=False,
        warning_count=warning_count,
    )


def ingest_file(
    path: Path,
    model: str | None = None,
    raw_root: Path = RAW_DIR,
    db_path: Path | None = None,
    repair: bool = False,
    repair_semantic: bool = False,
    raw_original_dir: Path = RAW_ORIGINAL_DIR,
    raw_repaired_dir: Path = RAW_REPAIRED_DIR,
    repair_log_dir: Path = REPAIR_LOG_DIR,
) -> int:
    result = _ingest_file_impl(
        path=path,
        model=model,
        raw_root=raw_root,
        db_path=db_path,
        repair=repair,
        repair_semantic=repair_semantic,
        raw_original_dir=raw_original_dir,
        raw_repaired_dir=raw_repaired_dir,
        repair_log_dir=repair_log_dir,
    )
    return result.records_ingested


def ingest_dir(
    dir_path: Path,
    model: str | None = None,
    raw_root: Path = RAW_DIR,
    db_path: Path | None = None,
    repair: bool = False,
    repair_semantic: bool = False,
    raw_original_dir: Path = RAW_ORIGINAL_DIR,
    raw_repaired_dir: Path = RAW_REPAIRED_DIR,
    repair_log_dir: Path = REPAIR_LOG_DIR,
) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        raise NotADirectoryError(f"Ingest source directory does not exist or is not a directory: {dir_path}")

    total = 0
    paths = sorted(dir_path.rglob("*.json")) + sorted(dir_path.rglob("*.jsonl"))
    for path in paths:
        total += _ingest_file_impl(
            path=path,
            model=model,
            raw_root=raw_root,
            db_path=db_path,
            repair=repair,
            repair_semantic=repair_semantic,
            raw_original_dir=raw_original_dir,
            raw_repaired_dir=raw_repaired_dir,
            repair_log_dir=repair_log_dir,
        ).records_ingested
    return total


def discover_raw_candidates(raw_dir: Path = RAW_DIR) -> tuple[list[Path], list[str], int]:
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw directory does not exist or is not a directory: {raw_dir}")

    candidates: list[Path] = []
    warnings: list[str] = []
    all_paths = sorted(raw_dir.rglob("*.json")) + sorted(raw_dir.rglob("*.jsonl"))
    for path in all_paths:
        rel = path.relative_to(raw_dir)
        if len(rel.parts) != 2:
            warnings.append(
                f"skip (layout): {rel} "
                "(expected raw/<model>/<YYYYMMDD(.json|.jsonl) or YYYYMMDD-N(.json|.jsonl)>)"
            )
            continue

        model = rel.parts[0].lower()
        if model not in MODEL_CHOICES:
            warnings.append(f"skip (model): {rel} (unknown model directory)")
            continue

        if extract_source_date(path) is None:
            warnings.append(
                f"skip (filename): {rel} "
                "(expected YYYYMMDD.json/.jsonl or YYYYMMDD-N.json/.jsonl, N>=1)"
            )
            continue

        candidates.append(path)

    return candidates, warnings, len(all_paths)


def ingest_raw_tree(
    raw_dir: Path = RAW_DIR,
    db_path: Path | None = None,
    repair: bool = False,
    repair_semantic: bool = False,
    raw_original_dir: Path = RAW_ORIGINAL_DIR,
    raw_repaired_dir: Path = RAW_REPAIRED_DIR,
    repair_log_dir: Path = REPAIR_LOG_DIR,
) -> IngestRawReport:
    if repair_semantic and not repair:
        raise ValueError("--repair-semantic requires --repair")

    candidates, warnings, scanned = discover_raw_candidates(raw_dir=raw_dir)
    ingested_files = 0
    skipped_duplicates = 0
    ingested_records = 0
    failed_records = 0
    original_saved = 0
    repaired_saved = 0
    repairs_applied = 0
    repair_failed_files = 0
    validation_failed_files = 0
    all_warnings = list(warnings)

    for path in candidates:
        try:
            result = _ingest_file_impl(
                path=path,
                model=None,
                raw_root=raw_dir,
                db_path=db_path,
                repair=repair,
                repair_semantic=repair_semantic,
                raw_original_dir=raw_original_dir,
                raw_repaired_dir=raw_repaired_dir,
                repair_log_dir=repair_log_dir,
            )
        except RepairValidationError as exc:
            validation_failed_files += 1
            failed_records += 1
            rel = path.relative_to(raw_dir)
            all_warnings.append(f"validation_failed: {rel} ({exc})")
            continue
        except RepairIngestError as exc:
            repair_failed_files += 1
            failed_records += 1
            rel = path.relative_to(raw_dir)
            all_warnings.append(f"repair_failed: {rel} ({exc})")
            continue
        except (RuntimeError, FileNotFoundError, NotADirectoryError, ValueError, sqlite3.DatabaseError) as exc:
            failed_records += 1
            rel = path.relative_to(raw_dir)
            all_warnings.append(f"failed: {rel} ({exc})")
            continue

        original_saved += result.original_saved
        repaired_saved += result.repaired_saved
        repairs_applied += result.repairs_applied

        if result.notes:
            all_warnings.extend(result.notes)

        if result.skipped_duplicate:
            skipped_duplicates += 1
            continue
        ingested_files += 1
        ingested_records += result.records_ingested

    warnings_count = len(all_warnings)
    return IngestRawReport(
        scanned_files=scanned,
        scanned_candidates=len(candidates),
        ingested_files=ingested_files,
        skipped_duplicates=skipped_duplicates,
        skipped_layout=len(warnings),
        ingested_records=ingested_records,
        failed_records=failed_records,
        warnings=tuple(all_warnings),
        original_saved=original_saved,
        repaired_saved=repaired_saved,
        repairs_applied=repairs_applied,
        warnings_count=warnings_count,
        repair_failed_files=repair_failed_files,
        validation_failed_files=validation_failed_files,
    )
