from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from arena.synthesis.db import ensure_schema
from arena.synthesis.normalization import normalize_optional_text
from arena.synthesis.paths import REVIEW_DIR
from arena.synthesis.proposition_review_shared import (
    ALLOWED_TRANSITIONS,
    HUMAN_LOCKED_STATUSES,
    PROPOSITION_REVIEW_STATUS_VOCAB,
    PropositionReviewStatusUpdateReport,
    ReviewQueueExportReport,
    connect_for_update,
    default_queue_path,
    now_iso,
    safe_json_loads_list,
)
from arena.synthesis.proposition_review_triage import write_jsonl


def _load_latest_triage_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
          t.id AS triage_record_id,
          t.proposition_id,
          t.review_status_suggested,
          t.validity_assessment,
          t.support_strength,
          t.decision_risk,
          t.review_priority_score,
          t.consensus_class,
          t.observation_metrics_json,
          t.key_issues_json,
          t.evidence_coverage_json,
          t.consistency_check_json,
          t.action_suggestion_json,
          t.rewrite_suggestion,
          t.next_data_needed_json,
          t.precheck_flags_json,
          t.source_provenance_json,
          t.created_at,
          p.question AS proposition_question,
          p.target AS proposition_target,
          COALESCE(s.current_status, 'pending') AS current_review_status
        FROM proposition_triage_records t
        JOIN (
          SELECT proposition_id, MAX(id) AS max_id
          FROM proposition_triage_records
          GROUP BY proposition_id
        ) latest
          ON latest.max_id = t.id
        JOIN propositions p
          ON p.proposition_id = t.proposition_id
        LEFT JOIN proposition_review_state s
          ON s.proposition_id = t.proposition_id
        ORDER BY t.review_priority_score DESC, t.decision_risk DESC, t.proposition_id
        """
    ).fetchall()


def _build_queue_rows(
    conn: sqlite3.Connection,
    *,
    limit: int | None,
    include_final: bool,
) -> list[dict[str, object]]:
    rows = _load_latest_triage_rows(conn)
    queue_rows: list[dict[str, object]] = []
    for row in rows:
        current_status = str(row["current_review_status"])
        if not include_final and current_status in HUMAN_LOCKED_STATUSES:
            continue
        observation_metrics = (
            json.loads(row["observation_metrics_json"]) if isinstance(row["observation_metrics_json"], str) else {}
        )
        unique_model_count = int(observation_metrics.get("unique_model_count", 0))
        same_model_repeat_count_max = int(observation_metrics.get("same_model_repeat_count_max", 0))
        contradicting_observation_count = int(observation_metrics.get("contradicting_observation_count", 0))
        item: dict[str, object] = {
            "triage_record_id": int(row["triage_record_id"]),
            "proposition_id": str(row["proposition_id"]),
            "proposition_question": str(row["proposition_question"]),
            "proposition_target": str(row["proposition_target"]),
            "suggested_review_status": str(row["review_status_suggested"]),
            "current_review_status": current_status,
            "validity_assessment": str(row["validity_assessment"]),
            "support_strength": float(row["support_strength"]),
            "decision_risk": float(row["decision_risk"]),
            "review_priority_score": float(row["review_priority_score"]),
            "consensus_class": str(row["consensus_class"]),
            "unique_model_count": unique_model_count,
            "same_model_repeat_count": same_model_repeat_count_max,
            "contradicting_observation_count": contradicting_observation_count,
            "key_issues": safe_json_loads_list(row["key_issues_json"]),
            "action_suggestion": json.loads(row["action_suggestion_json"]) if isinstance(row["action_suggestion_json"], str) else {},
            "created_at": str(row["created_at"]),
        }
        queue_rows.append(item)
        if limit is not None and len(queue_rows) >= limit:
            break
    return queue_rows


def export_review_queue(
    *,
    db_path: Path | None = None,
    output_path: Path | None = None,
    review_dir: Path = REVIEW_DIR,
    format_name: str = "jsonl",
    limit: int | None = None,
    include_final: bool = False,
) -> ReviewQueueExportReport:
    normalized_format = format_name.casefold().strip()
    if normalized_format not in {"jsonl", "csv"}:
        raise ValueError("format_name must be either 'jsonl' or 'csv'.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided.")

    target_output = output_path or default_queue_path(review_dir=review_dir, format_name=normalized_format)
    with connect_for_update(db_path) as conn:
        ensure_schema(conn)
        queue_rows = _build_queue_rows(conn, limit=limit, include_final=include_final)

    target_output.parent.mkdir(parents=True, exist_ok=True)
    if normalized_format == "jsonl":
        write_jsonl(target_output, rows=queue_rows)  # type: ignore[arg-type]
    else:
        with target_output.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "triage_record_id",
                    "proposition_id",
                    "proposition_question",
                    "proposition_target",
                    "suggested_review_status",
                    "current_review_status",
                    "validity_assessment",
                    "support_strength",
                    "decision_risk",
                    "review_priority_score",
                    "unique_model_count",
                    "same_model_repeat_count",
                    "contradicting_observation_count",
                    "consensus_class",
                    "key_issues",
                    "recommended_action",
                    "action_reason",
                    "created_at",
                ]
            )
            for row in queue_rows:
                action = row.get("action_suggestion", {})
                recommended_action = action.get("recommended_action") if isinstance(action, dict) else None
                action_reason = action.get("reason") if isinstance(action, dict) else None
                writer.writerow(
                    [
                        row["triage_record_id"],
                        row["proposition_id"],
                        row["proposition_question"],
                        row["proposition_target"],
                        row["suggested_review_status"],
                        row["current_review_status"],
                        row["validity_assessment"],
                        row["support_strength"],
                        row["decision_risk"],
                        row["review_priority_score"],
                        row["unique_model_count"],
                        row["same_model_repeat_count"],
                        row["contradicting_observation_count"],
                        row["consensus_class"],
                        "; ".join(str(item) for item in row.get("key_issues", [])),
                        recommended_action,
                        action_reason,
                        row["created_at"],
                    ]
                )

    return ReviewQueueExportReport(
        exported_rows=len(queue_rows),
        output_path=str(target_output),
        format=normalized_format,
        include_final=include_final,
    )


def _normalize_review_status(value: str | None, *, field_name: str) -> str:
    token = normalize_optional_text(value)
    if token is None:
        allowed = ", ".join(PROPOSITION_REVIEW_STATUS_VOCAB)
        raise ValueError(f"{field_name} must be one of: {allowed}.")
    lowered = token.casefold()
    if lowered not in PROPOSITION_REVIEW_STATUS_VOCAB:
        allowed = ", ".join(PROPOSITION_REVIEW_STATUS_VOCAB)
        raise ValueError(f"{field_name} must be one of: {allowed}.")
    return lowered


def _normalize_optional_review_status(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _normalize_review_status(value, field_name=field_name)


def _normalize_proposition_ids(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    for value in values:
        token = normalize_optional_text(value)
        if token is None:
            continue
        if token not in normalized:
            normalized.append(token)
    return normalized


def _is_transition_allowed(current_status: str, requested_status: str) -> bool:
    if current_status == requested_status:
        return True
    allowed = ALLOWED_TRANSITIONS.get(current_status, set())
    return requested_status in allowed


def _resolve_selector_ids(
    conn: sqlite3.Connection,
    *,
    proposition_ids: Sequence[str] | None,
    triage_record_ids: Sequence[int] | None,
) -> tuple[list[str], list[str]]:
    selected_ids = _normalize_proposition_ids(proposition_ids)
    missing: list[str] = []

    triage_ids = [int(value) for value in (triage_record_ids or [])]
    if triage_ids:
        placeholders = ", ".join("?" for _ in triage_ids)
        rows = conn.execute(
            f"""
            SELECT id, proposition_id
            FROM proposition_triage_records
            WHERE id IN ({placeholders})
            """,
            triage_ids,
        ).fetchall()
        found_ids = {int(row["id"]) for row in rows}
        for row in rows:
            proposition_id = str(row["proposition_id"])
            if proposition_id not in selected_ids:
                selected_ids.append(proposition_id)
        for triage_id in triage_ids:
            if triage_id not in found_ids:
                missing.append(f"triage_record:{triage_id}")

    return selected_ids, missing


def set_proposition_review_status(
    *,
    db_path: Path | None = None,
    proposition_ids: Sequence[str] | None = None,
    triage_record_ids: Sequence[int] | None = None,
    review_status: str,
    dry_run: bool = False,
    limit: int | None = None,
    where_current_status: str | None = None,
    allow_human_overwrite: bool = False,
    force: bool = False,
    reason: str | None = None,
) -> PropositionReviewStatusUpdateReport:
    requested_status = _normalize_review_status(review_status, field_name="review_status")
    normalized_where_status = _normalize_optional_review_status(where_current_status, field_name="where_current_status")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided.")
    if not proposition_ids and not triage_record_ids:
        raise ValueError("Specify at least one selector: --proposition-ids or --triage-record-ids.")

    reason_used = normalize_optional_text(reason)
    updated_ids: list[str] = []
    skipped_locked_ids: list[str] = []
    skipped_missing_ids: list[str] = []
    skipped_transition_ids: list[str] = []
    unchanged_records = 0

    with connect_for_update(db_path) as conn:
        ensure_schema(conn)
        selected_ids, selector_missing = _resolve_selector_ids(
            conn,
            proposition_ids=proposition_ids,
            triage_record_ids=triage_record_ids,
        )
        skipped_missing_ids.extend(selector_missing)
        if not selected_ids:
            return PropositionReviewStatusUpdateReport(
                requested_status=requested_status,
                matched_records=0,
                updated_records=0,
                unchanged_records=0,
                skipped_locked_records=0,
                skipped_missing_records=len(skipped_missing_ids),
                skipped_transition_records=0,
                history_records_written=0,
                reason_used=reason_used,
                dry_run=bool(dry_run),
                updated_ids=tuple(),
                skipped_locked_ids=tuple(),
                skipped_missing_ids=tuple(sorted(skipped_missing_ids)),
                skipped_transition_ids=tuple(),
            )

        placeholders = ", ".join("?" for _ in selected_ids)
        rows = conn.execute(
            f"""
            SELECT
              p.proposition_id,
              COALESCE(s.current_status, 'pending') AS current_status,
              COALESCE(s.triage_record_id, latest.max_id) AS triage_record_id
            FROM propositions p
            LEFT JOIN proposition_review_state s
              ON s.proposition_id = p.proposition_id
            LEFT JOIN (
              SELECT proposition_id, MAX(id) AS max_id
              FROM proposition_triage_records
              GROUP BY proposition_id
            ) latest
              ON latest.proposition_id = p.proposition_id
            WHERE p.proposition_id IN ({placeholders})
            ORDER BY p.proposition_id
            """,
            selected_ids,
        ).fetchall()

        found_ids = [str(row["proposition_id"]) for row in rows]
        found_set = set(found_ids)
        for proposition_id in selected_ids:
            if proposition_id not in found_set:
                skipped_missing_ids.append(proposition_id)

        candidate_rows = list(rows)
        if limit is not None:
            candidate_rows = candidate_rows[:limit]

        update_rows: list[tuple[str, str, int | None]] = []
        history_rows: list[tuple[str, int | None, str, str, str | None, int, str]] = []
        matched_rows: list[sqlite3.Row] = []

        for row in candidate_rows:
            proposition_id = str(row["proposition_id"])
            current_status = _normalize_review_status(str(row["current_status"]), field_name="current_status")
            triage_record_id = int(row["triage_record_id"]) if row["triage_record_id"] is not None else None
            if normalized_where_status is not None and current_status != normalized_where_status:
                continue

            matched_rows.append(row)
            if not allow_human_overwrite and current_status in HUMAN_LOCKED_STATUSES:
                skipped_locked_ids.append(proposition_id)
                continue
            if current_status == requested_status:
                unchanged_records += 1
                continue
            if not force and not _is_transition_allowed(current_status, requested_status):
                skipped_transition_ids.append(proposition_id)
                continue

            update_rows.append((proposition_id, requested_status, triage_record_id))
            updated_ids.append(proposition_id)
            history_rows.append(
                (
                    proposition_id,
                    triage_record_id,
                    current_status,
                    requested_status,
                    reason_used,
                    1 if (force or allow_human_overwrite) else 0,
                    now_iso(),
                )
            )

        if update_rows and not dry_run:
            timestamp = now_iso()
            for proposition_id, status_value, triage_record_id in update_rows:
                conn.execute(
                    """
                    INSERT INTO proposition_review_state (
                      proposition_id, current_status, triage_record_id, updated_at, note
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(proposition_id) DO UPDATE SET
                      current_status = excluded.current_status,
                      triage_record_id = excluded.triage_record_id,
                      updated_at = excluded.updated_at,
                      note = excluded.note
                    """,
                    (
                        proposition_id,
                        status_value,
                        triage_record_id,
                        timestamp,
                        reason_used,
                    ),
                )

            conn.executemany(
                """
                INSERT INTO proposition_review_history (
                  proposition_id,
                  triage_record_id,
                  old_status,
                  new_status,
                  change_reason,
                  force_applied,
                  changed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                history_rows,
            )
            conn.commit()
            history_written = len(history_rows)
        else:
            history_written = 0

    return PropositionReviewStatusUpdateReport(
        requested_status=requested_status,
        matched_records=len(matched_rows),
        updated_records=len(updated_ids),
        unchanged_records=unchanged_records,
        skipped_locked_records=len(skipped_locked_ids),
        skipped_missing_records=len(skipped_missing_ids),
        skipped_transition_records=len(skipped_transition_ids),
        history_records_written=history_written,
        reason_used=reason_used,
        dry_run=bool(dry_run),
        updated_ids=tuple(updated_ids),
        skipped_locked_ids=tuple(sorted(skipped_locked_ids)),
        skipped_missing_ids=tuple(sorted(skipped_missing_ids)),
        skipped_transition_ids=tuple(sorted(skipped_transition_ids)),
    )
