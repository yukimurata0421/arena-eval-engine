from __future__ import annotations

import sqlite3
from pathlib import Path

from arena.synthesis import db as synthesis_db


def _patch_db(monkeypatch, tmp_path: Path) -> Path:
    db_dir = tmp_path / "db"
    db_path = db_dir / "synthesis.sqlite3"
    monkeypatch.setattr(synthesis_db, "DB_DIR", db_dir)
    monkeypatch.setattr(synthesis_db, "DB_PATH", db_path)
    monkeypatch.setattr(synthesis_db, "legacy_db_paths_present", lambda: [])
    return db_path


def test_init_db_upgrades_legacy_hypotheses_schema(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                source_path TEXT NOT NULL,
                claim TEXT,
                basis TEXT,
                ingested_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ingested_files (
                source_path TEXT PRIMARY KEY,
                source_sha256 TEXT NOT NULL,
                model TEXT NOT NULL,
                source_date TEXT,
                record_count INTEGER NOT NULL,
                ingested_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

    synthesis_db.init_db()

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(claims)").fetchall()}
        table_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}

    assert {
        "raw_text",
        "evidence_refs",
        "priority_hint",
        "created_at",
        "topic",
        "topic_confidence",
        "topic_method",
        "topic_reason",
        "baseline_label",
        "baseline_type",
        "baseline_confidence",
        "baseline_method",
        "baseline_reason",
        "exploration_axes_json",
        "fixed_conditions_json",
        "variable_conditions_json",
        "validity_scope",
        "review_status",
    }.issubset(columns)
    assert "claim_required_files" in table_names
    assert "baseline_clusters" in table_names
    assert "claim_baseline_links" in table_names
    assert "propositions" in table_names
    assert "claim_propositions" in table_names
    assert "proposition_triage_records" in table_names
    assert "proposition_review_state" in table_names
    assert "proposition_review_history" in table_names


def test_connect_rejects_legacy_db_conflict(monkeypatch, tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db_path = db_dir / "synthesis.sqlite3"
    legacy = tmp_path / "synthesis.db"
    monkeypatch.setattr(synthesis_db, "DB_DIR", db_dir)
    monkeypatch.setattr(synthesis_db, "DB_PATH", db_path)
    monkeypatch.setattr(synthesis_db, "legacy_db_paths_present", lambda: [legacy])

    try:
        synthesis_db.connect()
    except RuntimeError as exc:
        assert "Legacy SYNTHESIS DB file(s) detected" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for legacy DB conflict")


def test_init_db_is_idempotent_and_creates_view(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)

    first = synthesis_db.init_db()
    second = synthesis_db.init_db()
    assert first == second == str(db_path)

    with sqlite3.connect(db_path) as conn:
        names = {
            (row[0], row[1]) for row in conn.execute("SELECT name, type FROM sqlite_master").fetchall()
        }
    assert ("claims", "table") in names
    assert ("hypotheses", "view") in names
    assert ("ingested_files", "table") in names
    assert ("propositions", "table") in names
    assert ("claim_propositions", "table") in names
    assert ("claim_propositions_view", "view") in names
    assert ("proposition_triage_records", "table") in names
    assert ("proposition_review_state", "table") in names
    assert ("proposition_review_history", "table") in names

    with sqlite3.connect(db_path) as conn:
        triage_columns = {row[1] for row in conn.execute("PRAGMA table_info(proposition_triage_records)").fetchall()}
    assert "consensus_class" in triage_columns
    assert "observation_metrics_json" in triage_columns


def test_init_db_upgrades_claim_propositions_with_missing_question_column(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE propositions (
                proposition_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                target TEXT NOT NULL,
                convergence TEXT NOT NULL,
                convergence_detail TEXT NOT NULL,
                resolved_type TEXT NOT NULL,
                caveat TEXT,
                exploration_axes TEXT,
                priority_hint TEXT NOT NULL,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE claim_propositions (
                claim_id TEXT NOT NULL,
                proposition_id TEXT NOT NULL,
                source_ai TEXT NOT NULL,
                PRIMARY KEY (claim_id, proposition_id, source_ai),
                FOREIGN KEY (proposition_id) REFERENCES propositions(proposition_id)
            )
            """
        )
        conn.commit()

    synthesis_db.init_db()

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(claim_propositions)").fetchall()}
        names = {
            (row[0], row[1]) for row in conn.execute("SELECT name, type FROM sqlite_master").fetchall()
        }

    assert "proposition_question" in columns
    assert ("claim_propositions_view", "view") in names


def test_init_db_upgrades_proposition_triage_records_with_consensus_columns(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE propositions (
                proposition_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                target TEXT NOT NULL,
                convergence TEXT NOT NULL,
                convergence_detail TEXT NOT NULL,
                resolved_type TEXT NOT NULL,
                caveat TEXT,
                exploration_axes TEXT,
                priority_hint TEXT NOT NULL,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE proposition_triage_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposition_id TEXT NOT NULL,
                triage_run_id TEXT NOT NULL,
                review_status_suggested TEXT NOT NULL,
                validity_assessment TEXT NOT NULL,
                support_strength REAL NOT NULL,
                decision_risk REAL NOT NULL,
                review_priority_score REAL NOT NULL,
                key_issues_json TEXT NOT NULL,
                evidence_coverage_json TEXT NOT NULL,
                consistency_check_json TEXT NOT NULL,
                action_suggestion_json TEXT NOT NULL,
                rewrite_suggestion TEXT,
                next_data_needed_json TEXT NOT NULL,
                precheck_flags_json TEXT NOT NULL,
                source_provenance_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (proposition_id, triage_run_id)
            )
            """
        )
        conn.commit()

    synthesis_db.init_db()

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(proposition_triage_records)").fetchall()}

    assert "consensus_class" in columns
    assert "observation_metrics_json" in columns
