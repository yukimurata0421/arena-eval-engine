from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arena.synthesis.db import connect, ensure_schema
from arena.synthesis.normalization import normalize_optional_text

REVIEW_STATUS_VOCAB: tuple[str, ...] = (
    "auto_unreviewed",
    "auto_high_confidence",
    "needs_review",
    "human_confirmed",
    "human_corrected",
)

_HUMAN_LOCKED_STATUSES = {"human_confirmed", "human_corrected"}


@dataclass(frozen=True, slots=True)
class ReviewStatusUpdateReport:
    requested_status: str | None
    matched_records: int
    updated_records: int
    unchanged_records: int
    skipped_locked_records: int
    skipped_missing_records: int
    history_records_written: int
    reason_used: str | None
    dry_run: bool
    updated_ids: tuple[int, ...]
    skipped_locked_ids: tuple[int, ...]
    skipped_missing_ids: tuple[int, ...]


def _connect_for_update(db_path: Path | None) -> sqlite3.Connection:
    if db_path is None:
        return connect()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _normalize_review_status(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if token == "":
        return None
    lowered = token.casefold()
    if lowered in {"null", "none"}:
        return None
    if lowered not in REVIEW_STATUS_VOCAB:
        allowed = ", ".join(["null", *list(REVIEW_STATUS_VOCAB)])
        raise ValueError(f"{field_name} must be one of: {allowed}.")
    return lowered


def _normalize_current_status(value: str | None) -> str | None:
    normalized = normalize_optional_text(value)
    if normalized is None:
        return None
    lowered = normalized.casefold()
    if lowered in {"null", "none"}:
        return None
    return lowered


def _normalize_reason(value: str | None) -> str | None:
    normalized = normalize_optional_text(value)
    return normalized


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _history_new_status_value(value: str | None) -> str:
    # history keeps a non-null "new_status" token even when review_status is set to SQL NULL.
    return "null" if value is None else str(value)


def write_review_status_history_batch(
    conn: sqlite3.Connection,
    *,
    rows: Sequence[dict[str, Any]],
) -> int:
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO review_status_history (
          hypothesis_id, old_status, new_status, changed_at, change_reason
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                int(row["hypothesis_id"]),
                row.get("old_status"),
                str(row["new_status"]),
                str(row["changed_at"]),
                row.get("change_reason"),
            )
            for row in rows
        ],
    )
    return len(rows)


def set_review_status(
    *,
    db_path: Path | None = None,
    ids: Sequence[int] | None = None,
    topic: str | None = None,
    baseline_cluster_id: int | None = None,
    review_status: str | None,
    dry_run: bool = False,
    limit: int | None = None,
    where_current_status: str | None = None,
    allow_human_overwrite: bool = False,
    reason: str | None = None,
) -> ReviewStatusUpdateReport:
    requested_status = _normalize_review_status(review_status, field_name="review_status")
    if review_status is None:
        raise ValueError("review_status is required.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided.")

    id_filter = [int(value) for value in (ids or [])]
    topic_filter = normalize_optional_text(topic)
    has_topic_filter = topic_filter is not None
    has_baseline_filter = baseline_cluster_id is not None
    if not id_filter and not has_topic_filter and not has_baseline_filter:
        raise ValueError("Specify at least one selector: --ids, --topic, or --baseline-cluster-id.")

    apply_where_status = where_current_status is not None
    normalized_where_status = (
        _normalize_review_status(where_current_status, field_name="where_current_status")
        if apply_where_status
        else None
    )

    updated_ids: list[int] = []
    skipped_locked_ids: list[int] = []
    skipped_missing_ids: list[int] = []
    unchanged_records = 0
    history_rows: list[dict[str, Any]] = []
    reason_used = _normalize_reason(reason)

    with _connect_for_update(db_path) as conn:
        ensure_schema(conn)

        if id_filter:
            placeholders = ", ".join("?" for _ in id_filter)
            existing_rows = conn.execute(
                f"SELECT id FROM claims WHERE id IN ({placeholders})",
                id_filter,
            ).fetchall()
            existing_ids = {int(row["id"]) for row in existing_rows}
            skipped_missing_ids = sorted(set(id_filter) - existing_ids)

        query_parts = ["SELECT DISTINCT h.id, h.review_status FROM claims h"]
        params: list[object] = []
        where_parts: list[str] = []

        if has_baseline_filter:
            query_parts.append("JOIN claim_baseline_links l ON l.claim_id = h.id")
            where_parts.append("l.baseline_cluster_id = ?")
            params.append(int(baseline_cluster_id))

        if id_filter:
            placeholders = ", ".join("?" for _ in id_filter)
            where_parts.append(f"h.id IN ({placeholders})")
            params.extend(id_filter)

        if has_topic_filter:
            where_parts.append("h.topic = ?")
            params.append(topic_filter)

        query_sql = " ".join(query_parts)
        if where_parts:
            query_sql += " WHERE " + " AND ".join(where_parts)
        query_sql += " ORDER BY h.id"
        if limit is not None:
            query_sql += " LIMIT ?"
            params.append(int(limit))

        rows = conn.execute(query_sql, params).fetchall()
        matched_rows: list[sqlite3.Row] = []
        for row in rows:
            current_status = _normalize_current_status(row["review_status"])
            if apply_where_status and current_status != normalized_where_status:
                continue
            matched_rows.append(row)

        for row in matched_rows:
            claim_id = int(row["id"])
            current_status = _normalize_current_status(row["review_status"])
            if not allow_human_overwrite and current_status in _HUMAN_LOCKED_STATUSES:
                skipped_locked_ids.append(claim_id)
                continue
            if current_status == requested_status:
                unchanged_records += 1
                continue
            updated_ids.append(claim_id)
            history_rows.append(
                {
                    "hypothesis_id": claim_id,
                    "old_status": current_status,
                    "new_status": _history_new_status_value(requested_status),
                    "changed_at": _now_iso(),
                    "change_reason": reason_used,
                }
            )

        if updated_ids and not dry_run:
            conn.executemany(
                "UPDATE claims SET review_status = ? WHERE id = ?",
                [(requested_status, claim_id) for claim_id in updated_ids],
            )
            history_written = write_review_status_history_batch(conn, rows=history_rows)
            conn.commit()
        else:
            history_written = 0

    return ReviewStatusUpdateReport(
        requested_status=requested_status,
        matched_records=len(matched_rows),
        updated_records=len(updated_ids),
        unchanged_records=unchanged_records,
        skipped_locked_records=len(skipped_locked_ids),
        skipped_missing_records=len(skipped_missing_ids),
        history_records_written=history_written,
        reason_used=reason_used,
        dry_run=bool(dry_run),
        updated_ids=tuple(updated_ids),
        skipped_locked_ids=tuple(sorted(skipped_locked_ids)),
        skipped_missing_ids=tuple(sorted(skipped_missing_ids)),
    )


def render_review_update_result(report: ReviewStatusUpdateReport) -> str:
    requested = report.requested_status if report.requested_status is not None else "null"
    reason_text = report.reason_used if report.reason_used is not None else "-"
    parts = [
        f"requested_status={requested}",
        f"matched_records={report.matched_records}",
        f"updated_records={report.updated_records}",
        f"unchanged_records={report.unchanged_records}",
        f"skipped_locked_records={report.skipped_locked_records}",
        f"skipped_missing_records={report.skipped_missing_records}",
        f"history_records_written={report.history_records_written}",
        f"reason_used={reason_text}",
        f"dry_run={int(report.dry_run)}",
        f"updated_ids={','.join(str(v) for v in report.updated_ids) if report.updated_ids else '-'}",
        f"skipped_locked_ids={','.join(str(v) for v in report.skipped_locked_ids) if report.skipped_locked_ids else '-'}",
        f"skipped_missing_ids={','.join(str(v) for v in report.skipped_missing_ids) if report.skipped_missing_ids else '-'}",
    ]
    return " ".join(parts)
