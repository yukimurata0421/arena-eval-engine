from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from arena.synthesis import db as synthesis_db
from arena.synthesis.report_baselines import (
    build_baseline_report,
    render_baseline_report_json,
    render_baseline_report_text,
)


def _patch_db(monkeypatch, tmp_path: Path) -> Path:
    db_dir = tmp_path / "db"
    db_path = db_dir / "synthesis.sqlite3"
    monkeypatch.setattr(synthesis_db, "DB_DIR", db_dir)
    monkeypatch.setattr(synthesis_db, "DB_PATH", db_path)
    monkeypatch.setattr(synthesis_db, "legacy_db_paths_present", lambda: [])
    return db_path


def _seed_report_fixture(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO hypotheses (
                id, model, source_path, claim, claim_type, basis, priority_hint, created_at,
                topic, baseline_label, baseline_type, baseline_confidence, review_status, ingested_at
            ) VALUES
            (1, 'gpt', 'raw/gpt/1.json', 'supported claim 1', 'supported', 'basis', 'medium', '2026-03-23T00:00:00+00:00',
             'gain_tuning', 'pre-filter', 'hardware', 'medium', 'needs_review', '2026-03-23T00:00:00+00:00'),
            (2, 'gpt', 'raw/gpt/2.json', 'negative claim 2', 'negative', 'basis', 'low', '2026-03-23T00:00:00+00:00',
             'gain_tuning', 'pre_filter', 'hardware', 'high', 'auto_unreviewed', '2026-03-23T00:00:00+00:00'),
            (3, 'gpt', 'raw/gpt/3.json', 'missing baseline high priority', 'supported', 'basis', 'high', '2026-03-23T00:00:00+00:00',
             'coverage_auc', NULL, NULL, NULL, 'auto_unreviewed', '2026-03-23T00:00:00+00:00'),
            (4, 'gpt', 'raw/gpt/4.json', 'human confirmed claim', 'supported', 'basis', 'high', '2026-03-23T00:00:00+00:00',
             NULL, 'gain_49_6', 'parameter', 'medium', 'human_confirmed', '2026-03-23T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO baseline_clusters (id, label, type, confidence, cluster_reason, created_at)
            VALUES
            (1, 'pre_filter', 'hardware', 'medium', 'fixture', '2026-03-23T00:00:00+00:00'),
            (2, 'gain_49_6', 'parameter', 'medium', 'fixture', '2026-03-23T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO claim_baseline_links (claim_id, baseline_cluster_id, link_confidence)
            VALUES
            (1, 1, 'medium'),
            (2, 1, 'high'),
            (4, 2, 'high')
            """
        )
        conn.execute(
            """
            INSERT INTO claim_required_files (claim_id, file_name, priority, reason, required_for)
            VALUES (3, 'important.csv', 'A', 'missing baseline check', 'review')
            """
        )
        conn.commit()


def test_build_baseline_report_detects_mixed_cluster(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_report_fixture(db_path)

    report = build_baseline_report(db_path=db_path, include_cluster_details=True)
    assert report["summary"]["total_mixed_clusters"] == 1
    assert report["mixed_clusters"]
    assert report["mixed_clusters"][0]["has_mixed_claim_types"] is True


def test_build_baseline_report_extracts_high_priority_missing(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_report_fixture(db_path)

    report = build_baseline_report(db_path=db_path)
    missing = report["high_priority_without_baseline"]
    assert len(missing) == 1
    assert missing[0]["hypothesis_id"] == 3


def test_build_baseline_report_topic_distribution(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_report_fixture(db_path)

    report = build_baseline_report(db_path=db_path)
    by_topic = {row["topic"]: row for row in report["topic_distribution"]}
    assert by_topic["gain_tuning"]["claim_count"] == 2
    assert by_topic["gain_tuning"]["mixed_cluster_count"] == 1
    assert by_topic["coverage_auc"]["high_priority_no_baseline_count"] == 1


def test_build_baseline_report_limit_applies(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_report_fixture(db_path)

    report = build_baseline_report(db_path=db_path, limit=1, include_cluster_details=True)
    assert len(report["topic_distribution"]) == 1
    assert len(report["mixed_clusters"]) <= 1
    assert len(report["cluster_details"]) <= 1


def test_build_baseline_report_is_read_only(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_report_fixture(db_path)

    with sqlite3.connect(db_path) as conn:
        before_clusters = conn.execute("SELECT COUNT(*) FROM baseline_clusters").fetchone()[0]
        before_links = conn.execute("SELECT COUNT(*) FROM claim_baseline_links").fetchone()[0]

    _ = build_baseline_report(db_path=db_path, include_cluster_details=True)

    with sqlite3.connect(db_path) as conn:
        after_clusters = conn.execute("SELECT COUNT(*) FROM baseline_clusters").fetchone()[0]
        after_links = conn.execute("SELECT COUNT(*) FROM claim_baseline_links").fetchone()[0]

    assert before_clusters == after_clusters
    assert before_links == after_links


def test_renderers_work(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_report_fixture(db_path)

    report = build_baseline_report(db_path=db_path, include_cluster_details=True)
    text = render_baseline_report_text(report)
    assert "Baseline Report" in text
    json_text = render_baseline_report_json(report)
    parsed = json.loads(json_text)
    assert "summary" in parsed

