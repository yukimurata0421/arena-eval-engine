from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arena.synthesis.db import connect, ensure_schema
from arena.synthesis.normalization import (
    normalize_created_at,
    normalize_evidence_refs,
    normalize_optional_string_list,
    normalize_optional_text,
    normalize_priority,
    normalize_priority_hint,
    normalize_raw_text,
    normalize_required_files,
)
from arena.synthesis.validator import LoadedRecord


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def ensure_tables(conn: sqlite3.Connection) -> None:
    ensure_schema(conn)


def connect_for_ingest(db_path: Path | None) -> sqlite3.Connection:
    if db_path is None:
        return connect()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _normalize_record(data: dict[str, Any], model: str, source_path: Path, source_date: str | None) -> dict[str, Any]:
    basis = data.get("basis")
    if basis is None:
        basis = data.get("basis_summary")
    basis = normalize_optional_text(basis)

    raw_text = normalize_raw_text(data.get("raw_text"))
    evidence_refs = normalize_evidence_refs(data.get("evidence_refs"))
    priority_hint = normalize_priority_hint(data.get("priority_hint"))
    topic = normalize_optional_text(data.get("topic"))
    if topic is None:
        topic = normalize_optional_text(data.get("primary_topic"))
    topic_confidence = normalize_optional_text(data.get("topic_confidence"))
    topic_method = normalize_optional_text(data.get("topic_method"))
    topic_reason = normalize_optional_text(data.get("topic_reason"))
    baseline_label = normalize_optional_text(data.get("baseline_label"))
    baseline_type = normalize_optional_text(data.get("baseline_type"))
    baseline_confidence = normalize_optional_text(data.get("baseline_confidence"))
    baseline_method = normalize_optional_text(data.get("baseline_method"))
    baseline_reason = normalize_optional_text(data.get("baseline_reason"))
    baseline_candidate = data.get("baseline_candidate")
    if isinstance(baseline_candidate, dict):
        if baseline_label is None:
            baseline_label = normalize_optional_text(baseline_candidate.get("label"))
        if baseline_type is None:
            baseline_type = normalize_optional_text(baseline_candidate.get("type"))
        if baseline_confidence is None:
            baseline_confidence = normalize_optional_text(baseline_candidate.get("confidence"))
        if baseline_reason is None:
            baseline_reason = normalize_optional_text(baseline_candidate.get("reason"))
    exploration_axes = normalize_optional_string_list(data.get("exploration_axes"))
    fixed_conditions = normalize_optional_string_list(data.get("fixed_conditions"))
    variable_conditions = normalize_optional_string_list(data.get("variable_conditions"))
    validity_scope = normalize_optional_text(data.get("validity_scope"))
    review_status = normalize_optional_text(data.get("review_status"))

    created_at = normalize_created_at(data.get("created_at"))
    if created_at is None:
        created_at = source_date

    required_files = normalize_required_files(data.get("required_files")) or []
    for item in required_files:
        item["priority"] = normalize_priority(item.get("priority"))

    return {
        "model": model,
        "source_path": str(source_path),
        "claim": normalize_optional_text(data.get("claim")),
        "claim_type": normalize_optional_text(data.get("claim_type")),
        "basis": basis,
        "raw_text": raw_text,
        "evidence_refs": json.dumps(evidence_refs, ensure_ascii=False) if evidence_refs is not None else None,
        "priority_hint": priority_hint,
        "created_at": created_at,
        "topic": topic,
        "topic_confidence": topic_confidence,
        "topic_method": topic_method,
        "topic_reason": topic_reason,
        "baseline_label": baseline_label,
        "baseline_type": baseline_type,
        "baseline_confidence": baseline_confidence,
        "baseline_method": baseline_method,
        "baseline_reason": baseline_reason,
        "exploration_axes_json": json.dumps(exploration_axes, ensure_ascii=False) if exploration_axes is not None else None,
        "fixed_conditions_json": json.dumps(fixed_conditions, ensure_ascii=False) if fixed_conditions is not None else None,
        "variable_conditions_json": json.dumps(variable_conditions, ensure_ascii=False) if variable_conditions is not None else None,
        "validity_scope": validity_scope,
        "review_status": review_status,
        "required_files": required_files,
        "ingested_at": now_iso(),
    }


def ingest_records(
    conn: sqlite3.Connection,
    records: list[LoadedRecord] | tuple[LoadedRecord, ...],
    *,
    source_path: Path,
    model: str,
    source_date: str | None,
) -> int:
    count = 0
    for loaded in records:
        row = _normalize_record(loaded.payload, model=model, source_path=source_path, source_date=source_date)
        cursor = conn.execute(
            """
            INSERT INTO claims (
                model, source_path, claim, claim_type, basis, raw_text,
                evidence_refs, priority_hint, created_at,
                topic, topic_confidence, topic_method, topic_reason,
                baseline_label, baseline_type, baseline_confidence, baseline_method, baseline_reason,
                exploration_axes_json, fixed_conditions_json, variable_conditions_json,
                validity_scope, review_status, ingested_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["model"],
                row["source_path"],
                row["claim"],
                row["claim_type"],
                row["basis"],
                row["raw_text"],
                row["evidence_refs"],
                row["priority_hint"],
                row["created_at"],
                row["topic"],
                row["topic_confidence"],
                row["topic_method"],
                row["topic_reason"],
                row["baseline_label"],
                row["baseline_type"],
                row["baseline_confidence"],
                row["baseline_method"],
                row["baseline_reason"],
                row["exploration_axes_json"],
                row["fixed_conditions_json"],
                row["variable_conditions_json"],
                row["validity_scope"],
                row["review_status"],
                row["ingested_at"],
            ),
        )
        claim_id = int(cursor.lastrowid)
        for required_file in row["required_files"]:
            conn.execute(
                """
                INSERT INTO claim_required_files (
                    claim_id, file_name, priority, reason, required_for
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    claim_id,
                    required_file.get("file_name"),
                    required_file.get("priority"),
                    required_file.get("reason"),
                    required_file.get("required_for"),
                ),
            )
        count += 1
    return count


def delete_existing_source_claims(conn: sqlite3.Connection, source_path: Path) -> None:
    conn.execute("DELETE FROM claims WHERE source_path = ?", (str(source_path),))


def upsert_ingested_file(
    conn: sqlite3.Connection,
    *,
    source_path: Path,
    source_hash: str,
    model: str,
    source_date: str | None,
    record_count: int,
) -> None:
    conn.execute(
        """
        INSERT INTO ingested_files (
            source_path, source_sha256, model, source_date, record_count, ingested_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            source_sha256 = excluded.source_sha256,
            model = excluded.model,
            source_date = excluded.source_date,
            record_count = excluded.record_count,
            ingested_at = excluded.ingested_at
        """,
        (str(source_path), source_hash, model, source_date, record_count, now_iso()),
    )
