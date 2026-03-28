from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from arena.synthesis import db as synthesis_db
from arena.synthesis.suggest_actions import (
    build_action_suggestions,
    render_action_suggestions_json,
    render_action_suggestions_text,
)


def _patch_db(monkeypatch, tmp_path: Path) -> Path:
    db_dir = tmp_path / "db"
    db_path = db_dir / "synthesis.sqlite3"
    monkeypatch.setattr(synthesis_db, "DB_DIR", db_dir)
    monkeypatch.setattr(synthesis_db, "DB_PATH", db_path)
    monkeypatch.setattr(synthesis_db, "legacy_db_paths_present", lambda: [])
    return db_path


def _seed_actions_fixture(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO hypotheses (
                id, model, source_path, claim, claim_type, basis, priority_hint, created_at,
                topic, baseline_label, baseline_type, baseline_confidence, review_status, ingested_at
            ) VALUES
            (1, 'gpt', 'raw/gpt/1.json', 'supported in pre_filter', 'supported', 'basis', 'high', '2026-03-23T00:00:00+00:00',
             'gain_tuning', 'pre_filter', 'hardware', 'medium', 'auto_unreviewed', '2026-03-23T00:00:00+00:00'),
            (2, 'gpt', 'raw/gpt/2.json', 'negative in pre_filter', 'negative', 'basis', 'low', '2026-03-23T00:00:00+00:00',
             'gain_tuning', 'pre_filter', 'hardware', 'medium', 'auto_unreviewed', '2026-03-23T00:00:00+00:00'),
            (3, 'gpt', 'raw/gpt/3.json', 'supported in post_filter', 'supported', 'basis', 'low', '2026-03-23T00:00:00+00:00',
             'gain_tuning', 'post_filter', 'hardware', 'medium', 'needs_review', '2026-03-23T00:00:00+00:00'),
            (4, 'gpt', 'raw/gpt/4.json', 'supported in gain_49_6', 'supported', 'basis', 'medium', '2026-03-23T00:00:00+00:00',
             'gain_tuning', 'gain_49_6', 'parameter', 'high', 'auto_unreviewed', '2026-03-23T00:00:00+00:00'),
            (5, 'gpt', 'raw/gpt/5.json', 'missing baseline high', 'supported', 'basis', 'high', '2026-03-23T00:00:00+00:00',
             'coverage_auc', NULL, NULL, NULL, 'auto_unreviewed', '2026-03-23T00:00:00+00:00'),
            (6, 'gpt', 'raw/gpt/6.json', 'weak singleton baseline', 'supported', 'basis', 'low', '2026-03-23T00:00:00+00:00',
             'coverage_auc', 'weak_single', 'parameter', 'low', 'auto_unreviewed', '2026-03-23T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO baseline_clusters (id, label, type, confidence, cluster_reason, created_at)
            VALUES
            (1, 'pre_filter', 'hardware', 'medium', 'fixture', '2026-03-23T00:00:00+00:00'),
            (2, 'post_filter', 'hardware', 'medium', 'fixture', '2026-03-23T00:00:00+00:00'),
            (3, 'gain_49_6', 'parameter', 'high', 'fixture', '2026-03-23T00:00:00+00:00'),
            (4, 'weak_single', 'parameter', 'low', 'fixture', '2026-03-23T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO claim_baseline_links (claim_id, baseline_cluster_id, link_confidence)
            VALUES
            (1, 1, 'high'),
            (2, 1, 'medium'),
            (3, 2, 'medium'),
            (4, 3, 'high'),
            (6, 4, 'low')
            """
        )
        conn.commit()


def _seed_sort_fixture(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO hypotheses (
                id, model, source_path, claim, claim_type, basis, priority_hint, created_at,
                topic, baseline_label, baseline_type, baseline_confidence, review_status, ingested_at
            ) VALUES
            (11, 'gpt', 'raw/gpt/11.json', 'high missing baseline reviewed', 'supported', 'basis', 'high', '2026-03-23T00:00:00+00:00',
             'gain_tuning', NULL, NULL, NULL, 'needs_review', '2026-03-23T00:00:00+00:00'),
            (12, 'gpt', 'raw/gpt/12.json', 'high missing baseline null topic', 'supported', 'basis', 'high', '2026-03-23T00:00:00+00:00',
             NULL, NULL, NULL, NULL, 'auto_unreviewed', '2026-03-23T00:00:00+00:00'),
            (13, 'gpt', 'raw/gpt/13.json', 'weak high-priority singleton', 'supported', 'basis', 'high', '2026-03-23T00:00:00+00:00',
             'coverage_auc', 'weak_high', 'parameter', 'low', 'auto_unreviewed', '2026-03-23T00:00:00+00:00'),
            (14, 'gpt', 'raw/gpt/14.json', 'weak low-priority singleton', 'supported', 'basis', 'low', '2026-03-23T00:00:00+00:00',
             'coverage_auc', 'weak_low', 'parameter', 'low', 'auto_unreviewed', '2026-03-23T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO baseline_clusters (id, label, type, confidence, cluster_reason, created_at)
            VALUES
            (21, 'weak_high', 'parameter', 'low', 'fixture', '2026-03-23T00:00:00+00:00'),
            (22, 'weak_low', 'parameter', 'low', 'fixture', '2026-03-23T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO claim_baseline_links (claim_id, baseline_cluster_id, link_confidence)
            VALUES
            (13, 21, 'low'),
            (14, 22, 'low')
            """
        )
        conn.commit()


def test_suggest_actions_detects_bottlenecks(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path)
    assert suggestions["bottlenecks"]
    row = suggestions["bottlenecks"][0]
    assert row["claim_id"] == 5
    assert row["severity"] == "high"
    assert isinstance(row["score"], int)
    assert row["recommended_next_step"]
    assert row["why_this_matters"]


def test_suggest_actions_detects_conflicts(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path)
    conflicts = suggestions["conflicts"]
    assert conflicts
    row = conflicts[0]
    assert row["cluster_id"] == 1
    assert sorted(row["claim_ids"]) == [1, 2]
    assert row["severity"] in {"high", "medium"}
    assert isinstance(row["score"], int)
    assert row["recommended_next_step"]


def test_suggest_actions_detects_fragmentation(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path)
    fragments = suggestions["fragmentation"]
    assert fragments
    row = fragments[0]
    assert row["topic"] == "gain_tuning"
    assert row["cluster_count"] == 3
    assert row["severity"] in {"medium", "low", "high"}
    assert isinstance(row["score"], int)


def test_suggest_actions_detects_weak_baselines(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path)
    weak = suggestions["weak_baselines"]
    assert weak
    row = weak[0]
    assert row["cluster_id"] == 4
    assert row["severity"] in {"low", "medium"}
    assert isinstance(row["score"], int)


def test_suggest_actions_json_output(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path, limit=2)
    text = render_action_suggestions_json(suggestions)
    parsed = json.loads(text)
    assert "bottlenecks" in parsed
    assert "conflicts" in parsed
    assert "fragmentation" in parsed
    assert "weak_baselines" in parsed
    assert "topic_health_summary" in parsed


def test_suggest_actions_limit_applies(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path, limit=1)
    assert len(suggestions["bottlenecks"]) <= 1
    assert len(suggestions["conflicts"]) <= 1
    assert len(suggestions["fragmentation"]) <= 1
    assert len(suggestions["weak_baselines"]) <= 1
    assert len(suggestions["topic_health_summary"]) <= 1


def test_suggest_actions_is_read_only(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    with sqlite3.connect(db_path) as conn:
        before_clusters = conn.execute("SELECT COUNT(*) FROM baseline_clusters").fetchone()[0]
        before_links = conn.execute("SELECT COUNT(*) FROM claim_baseline_links").fetchone()[0]

    _ = build_action_suggestions(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        after_clusters = conn.execute("SELECT COUNT(*) FROM baseline_clusters").fetchone()[0]
        after_links = conn.execute("SELECT COUNT(*) FROM claim_baseline_links").fetchone()[0]

    assert before_clusters == after_clusters
    assert before_links == after_links


def test_suggest_actions_includes_topic_health_summary(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path)
    assert suggestions["topic_health_summary"]
    row = suggestions["topic_health_summary"][0]
    assert "topic" in row
    assert "health_score" in row
    assert "health_status" in row


def test_suggest_actions_text_includes_severity_and_recommendation(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path)
    text = render_action_suggestions_text(suggestions)
    assert "[HIGH]" in text or "[MEDIUM]" in text or "[LOW]" in text
    assert "next_step=" in text
    assert "Topic Health Summary:" in text


def test_suggest_actions_filters_still_work(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    only_conflicts = build_action_suggestions(db_path=db_path, only_conflicts=True)
    assert only_conflicts["bottlenecks"] == []
    assert only_conflicts["fragmentation"] == []
    assert only_conflicts["weak_baselines"] == []
    assert only_conflicts["conflicts"]

    only_high = build_action_suggestions(db_path=db_path, only_high=True)
    assert only_high["bottlenecks"]
    assert only_high["conflicts"] == []
    assert only_high["fragmentation"] == []
    assert only_high["weak_baselines"] == []


def test_suggest_actions_min_severity_high(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path, min_severity="high")
    action_keys = ("bottlenecks", "conflicts", "fragmentation", "weak_baselines")
    for key in action_keys:
        for row in suggestions[key]:
            assert row["severity"] == "high"


def test_suggest_actions_min_severity_medium(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path, min_severity="medium")
    action_keys = ("bottlenecks", "conflicts", "fragmentation", "weak_baselines")
    severities = {row["severity"] for key in action_keys for row in suggestions[key]}
    assert "low" not in severities


def test_suggest_actions_sort_by_score(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_sort_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path, sort_by="score")
    weak = suggestions["weak_baselines"]
    assert len(weak) >= 2
    assert int(weak[0]["score"]) >= int(weak[1]["score"])


def test_suggest_actions_sort_by_severity(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_sort_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path, sort_by="severity")
    weak = suggestions["weak_baselines"]
    assert len(weak) >= 2
    rank = {"high": 3, "medium": 2, "low": 1}
    left = rank[str(weak[0]["severity"])]
    right = rank[str(weak[1]["severity"])]
    assert left >= right


def test_suggest_actions_json_and_text_reflect_filtering(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_actions_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path, min_severity="high", sort_by="score")
    parsed = json.loads(render_action_suggestions_json(suggestions))
    action_keys = ("bottlenecks", "conflicts", "fragmentation", "weak_baselines")
    for key in action_keys:
        for row in parsed[key]:
            assert row["severity"] == "high"

    text = render_action_suggestions_text(suggestions)
    assert "score=" in text


def test_suggest_actions_text_reflects_sort_order(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    _seed_sort_fixture(db_path)

    suggestions = build_action_suggestions(db_path=db_path, sort_by="score")
    text = render_action_suggestions_text(suggestions)
    first = text.find("cluster_id=21")
    second = text.find("cluster_id=22")
    assert first != -1 and second != -1
    assert first < second
