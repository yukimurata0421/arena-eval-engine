from __future__ import annotations

import sqlite3
from pathlib import Path

from arena.synthesis import db as synthesis_db
from arena.synthesis.review_status import render_review_update_result, set_review_status


def _patch_db(monkeypatch, tmp_path: Path) -> Path:
    db_dir = tmp_path / "db"
    db_path = db_dir / "synthesis.sqlite3"
    monkeypatch.setattr(synthesis_db, "DB_DIR", db_dir)
    monkeypatch.setattr(synthesis_db, "DB_PATH", db_path)
    monkeypatch.setattr(synthesis_db, "legacy_db_paths_present", lambda: [])
    return db_path


def _seed_review_fixture(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO hypotheses (
                id, model, source_path, claim, review_status, topic, baseline_label, ingested_at
            ) VALUES
            (1, 'gpt', 'raw/gpt/1.json', 'claim 1', 'needs_review', 'gain_tuning', 'pre_filter', '2026-03-23T00:00:00+00:00'),
            (2, 'gpt', 'raw/gpt/2.json', 'claim 2', 'auto_unreviewed', 'gain_tuning', NULL, '2026-03-23T00:00:00+00:00'),
            (3, 'gpt', 'raw/gpt/3.json', 'claim 3', 'human_confirmed', 'gain_tuning', 'pre_filter', '2026-03-23T00:00:00+00:00'),
            (4, 'gpt', 'raw/gpt/4.json', 'claim 4', 'human_corrected', 'coverage_auc', 'post_filter', '2026-03-23T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO baseline_clusters (id, label, type, confidence, cluster_reason, created_at)
            VALUES (9, 'pre_filter', 'hardware', 'medium', 'fixture', '2026-03-23T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO claim_baseline_links (claim_id, baseline_cluster_id, link_confidence)
            VALUES (1, 9, 'high'), (3, 9, 'high')
            """
        )
        conn.commit()


def _get_review_status(db_path: Path, claim_id: int) -> str | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT review_status FROM hypotheses WHERE id = ?", (claim_id,)).fetchone()
    return None if row is None else row[0]


def _get_history_rows(db_path: Path) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            """
            SELECT hypothesis_id, old_status, new_status, change_reason
            FROM review_status_history
            ORDER BY id
            """
        ).fetchall()


def test_review_status_history_table_exists_after_init(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='review_status_history'"
        ).fetchone()
    assert row is not None


def test_set_review_status_updates_ids(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_review_fixture(db_path)

    report = set_review_status(
        db_path=db_path,
        ids=[1, 2],
        review_status="human_confirmed",
    )
    assert report.updated_records == 2
    assert report.history_records_written == 2
    assert _get_review_status(db_path, 1) == "human_confirmed"
    assert _get_review_status(db_path, 2) == "human_confirmed"
    history = _get_history_rows(db_path)
    assert len(history) == 2


def test_set_review_status_dry_run_does_not_write(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_review_fixture(db_path)

    report = set_review_status(
        db_path=db_path,
        ids=[2],
        review_status="needs_review",
        dry_run=True,
    )
    assert report.updated_records == 1
    assert report.history_records_written == 0
    assert report.dry_run is True
    assert _get_review_status(db_path, 2) == "auto_unreviewed"
    assert _get_history_rows(db_path) == []


def test_set_review_status_no_change_writes_no_history(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_review_fixture(db_path)

    report = set_review_status(
        db_path=db_path,
        ids=[1],
        review_status="needs_review",
    )
    assert report.updated_records == 0
    assert report.unchanged_records == 1
    assert report.history_records_written == 0
    assert _get_history_rows(db_path) == []


def test_set_review_status_protects_human_locked_by_default(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_review_fixture(db_path)

    report = set_review_status(
        db_path=db_path,
        ids=[3, 4],
        review_status="needs_review",
    )
    assert report.updated_records == 0
    assert report.skipped_locked_records == 2
    assert report.history_records_written == 0
    assert _get_review_status(db_path, 3) == "human_confirmed"
    assert _get_review_status(db_path, 4) == "human_corrected"
    assert _get_history_rows(db_path) == []


def test_set_review_status_handles_missing_ids(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_review_fixture(db_path)

    report = set_review_status(
        db_path=db_path,
        ids=[999, 1],
        review_status="needs_review",
    )
    assert report.skipped_missing_records == 1
    assert report.skipped_missing_ids == (999,)
    assert report.matched_records == 1
    assert report.updated_records == 0
    assert report.history_records_written == 0


def test_set_review_status_supports_topic_and_cluster_filters(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_review_fixture(db_path)

    topic_report = set_review_status(
        db_path=db_path,
        topic="gain_tuning",
        review_status="needs_review",
        where_current_status="auto_unreviewed",
    )
    assert topic_report.updated_records == 1
    assert topic_report.history_records_written == 1
    assert _get_review_status(db_path, 2) == "needs_review"

    cluster_report = set_review_status(
        db_path=db_path,
        baseline_cluster_id=9,
        review_status="auto_unreviewed",
        allow_human_overwrite=True,
    )
    assert cluster_report.updated_records >= 1
    assert cluster_report.history_records_written >= 1


def test_set_review_status_reason_is_saved_when_provided(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_review_fixture(db_path)

    report = set_review_status(
        db_path=db_path,
        ids=[1],
        review_status="human_confirmed",
        reason="baseline confirmed after manual review",
    )
    assert report.updated_records == 1
    history = _get_history_rows(db_path)
    assert len(history) == 1
    assert history[0][3] == "baseline confirmed after manual review"


def test_set_review_status_reason_is_null_when_omitted(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_review_fixture(db_path)

    report = set_review_status(
        db_path=db_path,
        ids=[2],
        review_status="needs_review",
    )
    assert report.updated_records == 1
    history = _get_history_rows(db_path)
    assert len(history) == 1
    assert history[0][3] is None


def test_render_review_update_result_contains_summary(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_review_fixture(db_path)

    report = set_review_status(
        db_path=db_path,
        ids=[1],
        review_status="human_corrected",
        dry_run=True,
    )
    text = render_review_update_result(report)
    assert "requested_status=human_corrected" in text
    assert "matched_records=1" in text
    assert "history_records_written=0" in text
    assert "unchanged_records=0" in text
    assert "dry_run=1" in text
