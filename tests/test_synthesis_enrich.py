from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from arena.synthesis import db as synthesis_db
from arena.synthesis.enrich import enrich_claims
from arena.synthesis.ingest import ingest_raw_tree


def _patch_db(monkeypatch, tmp_path: Path) -> Path:
    db_dir = tmp_path / "db"
    db_path = db_dir / "synthesis.sqlite3"
    monkeypatch.setattr(synthesis_db, "DB_DIR", db_dir)
    monkeypatch.setattr(synthesis_db, "DB_PATH", db_path)
    monkeypatch.setattr(synthesis_db, "legacy_db_paths_present", lambda: [])
    return db_path


def _base_record(claim: str, basis_summary: str = "Observed in experiment A.") -> dict[str, object]:
    return {
        "claim": claim,
        "claim_type": "supported",
        "basis_summary": basis_summary,
        "evidence_files": ["metrics.csv"],
        "metrics_used": [
            {
                "file": "metrics.csv",
                "metric": "auc",
                "value": 0.91,
                "context": {"series_name": "auc_trend", "condition": "test_split"},
            }
        ],
        "evidence_level": "direct",
        "limitation_or_counterpoint": None,
        "next_data_needed": "Long-term drift analysis.",
    }


def _ingest_one_record(tmp_path: Path, claim: str, basis_summary: str = "Observed in experiment A.") -> None:
    raw = tmp_path / "raw"
    (raw / "claude").mkdir(parents=True)
    (raw / "claude" / "20260322.json").write_text(
        json.dumps([_base_record(claim, basis_summary=basis_summary)], ensure_ascii=False),
        encoding="utf-8",
    )
    report = ingest_raw_tree(raw_dir=raw)
    assert report.ingested_files == 1


def test_enrich_topic_assignment_from_keywords(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _ingest_one_record(
        tmp_path,
        claim="Receiver gain=49.6 compared to baseline and AUC improved.",
    )

    report = enrich_claims(only_unreviewed=True)
    assert report.updated_claims == 1
    assert report.topic_assigned == 1

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT topic, topic_confidence, topic_method, topic_reason FROM hypotheses"
        ).fetchone()

    assert row is not None
    assert row[0] == "gain_tuning"
    assert row[1] in {"medium", "high"}
    assert row[2] == "rule"
    assert isinstance(row[3], str)
    assert row[3]


def test_enrich_unknown_or_null_when_unclassifiable(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _ingest_one_record(
        tmp_path,
        claim="This statement is generic and does not mention any known axis.",
    )

    _ = enrich_claims(only_unreviewed=True)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT topic FROM hypotheses").fetchone()

    assert row is not None
    assert row[0] in (None, "unknown")


def test_enrich_sets_exploration_axes(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _ingest_one_record(
        tmp_path,
        claim="Receiver gain increased and coverage AUC improved by 5%.",
    )

    _ = enrich_claims(only_unreviewed=True)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT exploration_axes_json FROM hypotheses").fetchone()

    assert row is not None
    axes = json.loads(row[0]) if row[0] else []
    assert "gain" in axes
    assert "coverage_auc" in axes


def test_enrich_baseline_from_explicit_comparison(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _ingest_one_record(
        tmp_path,
        claim="Compared to baseline gain=49.6, AUC improved after receiver tuning.",
    )

    report = enrich_claims(only_unreviewed=True)
    assert report.baseline_assigned == 1

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT baseline_label, baseline_type, baseline_confidence, baseline_method, baseline_reason
            FROM hypotheses
            """
        ).fetchone()

    assert row is not None
    assert row[0] == "gain_49_6"
    assert row[1] == "parameter"
    assert row[2] in {"high", "medium"}
    assert row[3] == "rule"
    assert isinstance(row[4], str)
    assert row[4]


def test_enrich_keeps_weak_baseline_as_null(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _ingest_one_record(
        tmp_path,
        claim="AUC improved by 1% in the latest run.",
    )

    _ = enrich_claims(only_unreviewed=True)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT baseline_label FROM hypotheses").fetchone()

    assert row is not None
    assert row[0] is None


def test_enrich_skips_human_locked_status(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _ingest_one_record(
        tmp_path,
        claim="Receiver gain=49.6 compared to baseline.",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE hypotheses SET review_status = 'human_confirmed', topic = 'data_quality'")
        conn.commit()

    report = enrich_claims(only_unreviewed=False)
    assert report.skipped_human_locked == 1
    assert report.updated_claims == 0

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT review_status, topic FROM hypotheses").fetchone()

    assert row is not None
    assert row[0] == "human_confirmed"
    assert row[1] == "data_quality"


def test_enrich_dry_run_does_not_persist(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _ingest_one_record(
        tmp_path,
        claim="Receiver gain=49.6 compared to baseline and AUC improved.",
    )

    report = enrich_claims(dry_run=True, only_unreviewed=True)
    assert report.updated_claims == 1
    assert report.dry_run is True

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT topic, baseline_label, review_status FROM hypotheses").fetchone()

    assert row is not None
    assert row[0] is None
    assert row[1] is None
    assert row[2] is None


def test_enrich_topic_only_and_baseline_only(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _ingest_one_record(
        tmp_path,
        claim="Compared to baseline gain=49.6, receiver gain tuning improved AUC.",
    )

    topic_only_report = enrich_claims(topic_only=True, only_unreviewed=True)
    assert topic_only_report.updated_claims == 1

    with sqlite3.connect(db_path) as conn:
        row_topic = conn.execute("SELECT topic, baseline_label FROM hypotheses").fetchone()
    assert row_topic is not None
    assert row_topic[0] in {"gain_tuning", "coverage_auc"}
    assert row_topic[1] is None

    baseline_only_report = enrich_claims(baseline_only=True, only_unreviewed=True)
    assert baseline_only_report.updated_claims >= 1

    with sqlite3.connect(db_path) as conn:
        row_baseline = conn.execute("SELECT topic, baseline_label FROM hypotheses").fetchone()
    assert row_baseline is not None
    assert row_baseline[0] in {"gain_tuning", "coverage_auc"}
    assert row_baseline[1] == "gain_49_6"

