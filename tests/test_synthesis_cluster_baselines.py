from __future__ import annotations

import sqlite3
from pathlib import Path

from arena.synthesis import db as synthesis_db
from arena.synthesis.cluster_baselines import cluster_baselines, normalize_baseline_label


def _patch_db(monkeypatch, tmp_path: Path) -> Path:
    db_dir = tmp_path / "db"
    db_path = db_dir / "synthesis.sqlite3"
    monkeypatch.setattr(synthesis_db, "DB_DIR", db_dir)
    monkeypatch.setattr(synthesis_db, "DB_PATH", db_path)
    monkeypatch.setattr(synthesis_db, "legacy_db_paths_present", lambda: [])
    return db_path


def _insert_hypothesis(
    conn: sqlite3.Connection,
    *,
    source_suffix: str,
    baseline_label: str | None,
    baseline_type: str | None = None,
    baseline_confidence: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO hypotheses (
            model, source_path, claim, basis, ingested_at,
            baseline_label, baseline_type, baseline_confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "gpt",
            f"workspace/synthesis/raw/gpt/{source_suffix}.json",
            "Compared to baseline, AUC improved.",
            "Synthetic basis",
            "2026-03-23T00:00:00+00:00",
            baseline_label,
            baseline_type,
            baseline_confidence,
        ),
    )


def test_normalize_baseline_label_applies_rules() -> None:
    assert normalize_baseline_label("Pre-Filter") == "pre_filter"
    assert normalize_baseline_label("pre.filter") == "pre_filter"
    assert normalize_baseline_label("  gain 49.6 ") == "gain49_6"


def test_cluster_baselines_groups_same_normalized_label(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    with sqlite3.connect(db_path) as conn:
        _insert_hypothesis(conn, source_suffix="a", baseline_label="pre-filter", baseline_type="hardware", baseline_confidence="medium")
        _insert_hypothesis(conn, source_suffix="b", baseline_label="pre_filter", baseline_type="hardware", baseline_confidence="high")
        conn.commit()

    report = cluster_baselines(db_path=db_path)
    assert report.cluster_count == 1

    with sqlite3.connect(db_path) as conn:
        cluster_count = conn.execute("SELECT COUNT(*) FROM baseline_clusters").fetchone()[0]
        link_count = conn.execute("SELECT COUNT(*) FROM claim_baseline_links").fetchone()[0]

    assert cluster_count == 1
    assert link_count == 2


def test_cluster_baselines_confidence_rule(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    with sqlite3.connect(db_path) as conn:
        _insert_hypothesis(conn, source_suffix="a", baseline_label="gain_49_6", baseline_type="parameter", baseline_confidence="medium")
        _insert_hypothesis(conn, source_suffix="b", baseline_label="gain-49-6", baseline_type="parameter", baseline_confidence="high")
        _insert_hypothesis(conn, source_suffix="c", baseline_label="gain.49.6", baseline_type="parameter", baseline_confidence="medium")
        conn.commit()

    report = cluster_baselines(db_path=db_path)
    assert report.high_count == 1
    assert report.medium_count == 0
    assert report.low_count == 0

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT confidence FROM baseline_clusters").fetchone()
    assert row is not None
    assert row[0] == "high"


def test_cluster_baselines_claim_links_are_created(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    with sqlite3.connect(db_path) as conn:
        _insert_hypothesis(conn, source_suffix="a", baseline_label="post_cable_swap", baseline_type="hardware", baseline_confidence="high")
        conn.commit()

    report = cluster_baselines(db_path=db_path)
    assert report.claim_link_count == 1

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT l.claim_id, l.baseline_cluster_id, l.link_confidence
            FROM claim_baseline_links l
            """
        ).fetchone()
    assert row is not None
    assert row[2] == "high"


def test_cluster_baselines_dry_run_does_not_modify_db(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    with sqlite3.connect(db_path) as conn:
        _insert_hypothesis(conn, source_suffix="a", baseline_label="reference_window_2026_03", baseline_type="temporal", baseline_confidence="medium")
        conn.commit()

    report = cluster_baselines(db_path=db_path, dry_run=True)
    assert report.cluster_count == 1
    assert report.claim_link_count == 1
    assert report.dry_run is True

    with sqlite3.connect(db_path) as conn:
        cluster_count = conn.execute("SELECT COUNT(*) FROM baseline_clusters").fetchone()[0]
        link_count = conn.execute("SELECT COUNT(*) FROM claim_baseline_links").fetchone()[0]
    assert cluster_count == 0
    assert link_count == 0

