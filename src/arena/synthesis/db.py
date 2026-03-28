from __future__ import annotations

import sqlite3

from arena.synthesis.paths import DB_DIR, DB_PATH, legacy_db_paths_present
from arena.synthesis.schema import (
    BASELINE_CLUSTERS_TABLE_SQL,
    CLAIM_BASELINE_LINKS_INDEX_SQL,
    CLAIM_BASELINE_LINKS_TABLE_SQL,
    CLAIM_PROPOSITIONS_INDEX_SQL,
    CLAIM_PROPOSITIONS_TABLE_SQL,
    CLAIM_PROPOSITIONS_VIEW_SQL,
    CLAIM_REQUIRED_FILES_INDEX_SQL,
    CLAIM_REQUIRED_FILES_TABLE_SQL,
    CLAIMS_TABLE_SQL,
    HYPOTHESES_COMPAT_DELETE_TRIGGER_SQL,
    HYPOTHESES_COMPAT_INSERT_TRIGGER_SQL,
    HYPOTHESES_COMPAT_UPDATE_TRIGGER_SQL,
    HYPOTHESES_COMPAT_VIEW_SQL,
    INGESTED_FILES_TABLE_SQL,
    PROPOSITION_REVIEW_HISTORY_INDEX_SQL,
    PROPOSITION_REVIEW_HISTORY_TABLE_SQL,
    PROPOSITION_REVIEW_STATE_INDEX_SQL,
    PROPOSITION_REVIEW_STATE_TABLE_SQL,
    PROPOSITION_TRIAGE_RECORDS_PRIORITY_INDEX_SQL,
    PROPOSITION_TRIAGE_RECORDS_PROP_INDEX_SQL,
    PROPOSITION_TRIAGE_RECORDS_TABLE_SQL,
    PROPOSITIONS_TABLE_SQL,
    REVIEW_STATUS_HISTORY_INDEX_SQL,
    REVIEW_STATUS_HISTORY_TABLE_SQL,
)

# DB invariants (do not change silently):
# - DB target is always paths.DB_PATH (workspace/synthesis/db/synthesis.sqlite3)
# - legacy DB file presence is treated as hard error to avoid ambiguity

_REQUIRED_CLAIMS_COLUMNS: dict[str, str] = {
    "claim_type": "TEXT",
    "raw_text": "TEXT",
    "evidence_refs": "TEXT",
    "priority_hint": "TEXT",
    "created_at": "TEXT",
    "topic": "TEXT",
    "topic_confidence": "TEXT",
    "topic_method": "TEXT",
    "topic_reason": "TEXT",
    "baseline_label": "TEXT",
    "baseline_type": "TEXT",
    "baseline_confidence": "TEXT",
    "baseline_method": "TEXT",
    "baseline_reason": "TEXT",
    "exploration_axes_json": "TEXT",
    "fixed_conditions_json": "TEXT",
    "variable_conditions_json": "TEXT",
    "validity_scope": "TEXT",
    "review_status": "TEXT",
}

_REQUIRED_CLAIM_PROPOSITIONS_COLUMNS: dict[str, str] = {
    "proposition_question": "TEXT",
}

_REQUIRED_PROPOSITION_REVIEW_STATE_COLUMNS: dict[str, str] = {
    "triage_record_id": "INTEGER",
    "updated_at": "TEXT",
    "note": "TEXT",
}

_REQUIRED_PROPOSITION_TRIAGE_RECORDS_COLUMNS: dict[str, str] = {
    "consensus_class": "TEXT NOT NULL DEFAULT 'weak_sparse'",
    "observation_metrics_json": "TEXT NOT NULL DEFAULT '{}'",
}


def ensure_db_parent() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)


def _guard_legacy_db_conflicts() -> None:
    legacy_paths = legacy_db_paths_present()
    if not legacy_paths:
        return

    listed = ", ".join(str(path) for path in legacy_paths)
    raise RuntimeError(
        f"Legacy SYNTHESIS DB file(s) detected: {listed}. Canonical DB path is {DB_PATH}. "
        "Keep only the canonical DB filename to avoid ambiguity."
    )


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _sqlite_object_type(conn: sqlite3.Connection, object_name: str) -> str | None:
    row = conn.execute(
        "SELECT type FROM sqlite_master WHERE name = ? ORDER BY type DESC LIMIT 1",
        (object_name,),
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def _migrate_hypotheses_table_to_claims(conn: sqlite3.Connection) -> None:
    legacy_type = _sqlite_object_type(conn, "hypotheses")
    claims_type = _sqlite_object_type(conn, "claims")

    if claims_type == "table" and legacy_type == "table":
        raise RuntimeError(
            "Both legacy 'hypotheses' table and canonical 'claims' table exist. "
            "Please keep only one canonical claims table."
        )

    if claims_type is None and legacy_type == "table":
        conn.execute("ALTER TABLE hypotheses RENAME TO claims")


def _ensure_claims_columns(conn: sqlite3.Connection) -> None:
    existing = _table_columns(conn, "claims")
    for column, ddl_type in _REQUIRED_CLAIMS_COLUMNS.items():
        if column in existing:
            continue
        conn.execute(f"ALTER TABLE claims ADD COLUMN {column} {ddl_type}")


def _ensure_claim_propositions_columns(conn: sqlite3.Connection) -> None:
    existing = _table_columns(conn, "claim_propositions")
    for column, ddl_type in _REQUIRED_CLAIM_PROPOSITIONS_COLUMNS.items():
        if column in existing:
            continue
        conn.execute(f"ALTER TABLE claim_propositions ADD COLUMN {column} {ddl_type}")


def _ensure_proposition_review_state_columns(conn: sqlite3.Connection) -> None:
    existing = _table_columns(conn, "proposition_review_state")
    for column, ddl_type in _REQUIRED_PROPOSITION_REVIEW_STATE_COLUMNS.items():
        if column in existing:
            continue
        conn.execute(f"ALTER TABLE proposition_review_state ADD COLUMN {column} {ddl_type}")


def _ensure_proposition_triage_records_columns(conn: sqlite3.Connection) -> None:
    existing = _table_columns(conn, "proposition_triage_records")
    for column, ddl_type in _REQUIRED_PROPOSITION_TRIAGE_RECORDS_COLUMNS.items():
        if column in existing:
            continue
        conn.execute(f"ALTER TABLE proposition_triage_records ADD COLUMN {column} {ddl_type}")


def _ensure_hypotheses_compat_layer(conn: sqlite3.Connection) -> None:
    legacy_type = _sqlite_object_type(conn, "hypotheses")
    if legacy_type == "table":
        return
    conn.execute(HYPOTHESES_COMPAT_VIEW_SQL)
    conn.execute(HYPOTHESES_COMPAT_INSERT_TRIGGER_SQL)
    conn.execute(HYPOTHESES_COMPAT_UPDATE_TRIGGER_SQL)
    conn.execute(HYPOTHESES_COMPAT_DELETE_TRIGGER_SQL)


def ensure_schema(conn: sqlite3.Connection) -> None:
    _migrate_hypotheses_table_to_claims(conn)
    conn.execute(CLAIMS_TABLE_SQL)
    _ensure_claims_columns(conn)
    conn.execute(CLAIM_REQUIRED_FILES_TABLE_SQL)
    conn.execute(CLAIM_REQUIRED_FILES_INDEX_SQL)
    conn.execute(INGESTED_FILES_TABLE_SQL)
    conn.execute(BASELINE_CLUSTERS_TABLE_SQL)
    conn.execute(CLAIM_BASELINE_LINKS_TABLE_SQL)
    conn.execute(CLAIM_BASELINE_LINKS_INDEX_SQL)
    conn.execute(REVIEW_STATUS_HISTORY_TABLE_SQL)
    conn.execute(REVIEW_STATUS_HISTORY_INDEX_SQL)
    conn.execute(PROPOSITIONS_TABLE_SQL)
    conn.execute(CLAIM_PROPOSITIONS_TABLE_SQL)
    _ensure_claim_propositions_columns(conn)
    conn.execute(CLAIM_PROPOSITIONS_INDEX_SQL)
    conn.execute(CLAIM_PROPOSITIONS_VIEW_SQL)
    conn.execute(PROPOSITION_TRIAGE_RECORDS_TABLE_SQL)
    _ensure_proposition_triage_records_columns(conn)
    conn.execute(PROPOSITION_TRIAGE_RECORDS_PROP_INDEX_SQL)
    conn.execute(PROPOSITION_TRIAGE_RECORDS_PRIORITY_INDEX_SQL)
    conn.execute(PROPOSITION_REVIEW_STATE_TABLE_SQL)
    _ensure_proposition_review_state_columns(conn)
    conn.execute(PROPOSITION_REVIEW_STATE_INDEX_SQL)
    conn.execute(PROPOSITION_REVIEW_HISTORY_TABLE_SQL)
    conn.execute(PROPOSITION_REVIEW_HISTORY_INDEX_SQL)
    _ensure_hypotheses_compat_layer(conn)


def connect() -> sqlite3.Connection:
    ensure_db_parent()
    _guard_legacy_db_conflicts()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> str:
    with connect() as conn:
        ensure_schema(conn)
        conn.commit()
    return str(DB_PATH)
