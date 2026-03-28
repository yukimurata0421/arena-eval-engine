from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from arena.synthesis.db import ensure_schema
from arena.synthesis.proposition_layer import build_proposition_layer
from arena.synthesis.proposition_review import (
    export_review_queue,
    render_proposition_review_update_result,
    set_proposition_review_status,
    triage_propositions,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_propositions(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_dir = tmp_path / "raw"
    enriched_dir = tmp_path / "enriched"
    review_dir = tmp_path / "review"
    db_path = tmp_path / "db" / "synthesis.sqlite3"
    _write_json(
        raw_dir / "claude" / "20260327.json",
        [
            {
                "id": "CLAIM-001",
                "claim_type": "supported",
                "claim": "The auc_n_used change point is around 2026-01-10.",
                "basis_summary": "change point report",
                "evidence_files": ["change_point_report.csv"],
                "evidence_refs": ["CP-1"],
                "metrics_used": [{"file": "change_point_report.csv", "metric": "auc_n_used", "value": 1.2, "context": {"series_name": "daily", "condition": "baseline"}}],
                "evidence_level": "direct",
                "limitation_or_counterpoint": "Out-of-window behavior is unverified.",
                "next_data_needed": "Revalidate with additional windows.",
                "exploration_axes": ["coverage_auc", "baseline"],
            },
            {
                "id": "CLAIM-010",
                "claim_type": "future",
                "claim": "beta_minutes will definitely increase.",
                "basis_summary": "weak extrapolation",
                "evidence_files": [],
                "evidence_refs": [],
                "metrics_used": [],
                "evidence_level": "inferred",
                "limitation_or_counterpoint": None,
                "next_data_needed": "Additional samples.",
                "exploration_axes": ["stats"],
            },
        ],
    )
    build_proposition_layer(raw_dir=raw_dir, enriched_dir=enriched_dir, db_path=db_path)
    return db_path, enriched_dir, review_dir


def test_triage_propositions_persists_records_and_exports_queue(tmp_path: Path) -> None:
    db_path, enriched_dir, review_dir = _seed_propositions(tmp_path)

    triage = triage_propositions(
        db_path=db_path,
        enriched_dir=enriched_dir,
        review_dir=review_dir,
        triage_run_id="RUN-001",
    )
    assert triage.scanned_propositions == 11
    assert triage.triage_records_written == 11
    assert Path(triage.output_path).exists()

    with sqlite3.connect(db_path) as conn:
        triage_count = conn.execute("SELECT COUNT(*) FROM proposition_triage_records").fetchone()[0]
    assert triage_count == 11

    queue = export_review_queue(
        db_path=db_path,
        review_dir=review_dir,
        format_name="jsonl",
        limit=5,
    )
    assert queue.exported_rows >= 1
    queue_path = Path(queue.output_path)
    assert queue_path.exists()
    first_line = queue_path.read_text(encoding="utf-8").splitlines()[0]
    first_payload = json.loads(first_line)
    assert "proposition_id" in first_payload
    assert "review_priority_score" in first_payload
    assert "unique_model_count" in first_payload
    assert "same_model_repeat_count" in first_payload
    assert "contradicting_observation_count" in first_payload
    assert "consensus_class" in first_payload


def test_triage_uses_source_file_plus_claim_id_observation_key(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    enriched_dir = tmp_path / "enriched"
    review_dir = tmp_path / "review"
    db_path = tmp_path / "db" / "synthesis.sqlite3"

    _write_json(
        raw_dir / "claude" / "20260327.json",
        [
            {
                "id": "CLAIM-DUP",
                "claim_type": "supported",
                "claim": "The change point in auc_n_used is around 2026-01-10.",
                "basis_summary": "first observation",
                "evidence_files": [],
                "evidence_refs": [],
                "metrics_used": [],
                "evidence_level": "inferred",
                "limitation_or_counterpoint": "initial sample is small",
                "next_data_needed": "more windows",
            }
        ],
    )
    _write_json(
        raw_dir / "claude" / "20260328.json",
        [
            {
                "id": "CLAIM-DUP",
                "claim_type": "supported",
                "claim": "The change point in auc_n_used is around 2026-01-10.",
                "basis_summary": "second observation",
                "evidence_files": ["change_point_report.csv"],
                "evidence_refs": ["CP-2"],
                "metrics_used": [{"file": "change_point_report.csv", "metric": "auc_n_used", "value": 1.0, "context": {"series_name": "daily", "condition": "baseline"}}],
                "evidence_level": "direct",
                "limitation_or_counterpoint": "seasonality remains",
                "next_data_needed": "cross-check",
            }
        ],
    )

    build_proposition_layer(raw_dir=raw_dir, enriched_dir=enriched_dir, db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        collapsed = conn.execute(
            """
            SELECT COUNT(*)
            FROM claim_propositions
            WHERE source_ai = 'claude' AND claim_id = 'CLAIM-DUP' AND proposition_id = 'PROP-001'
            """
        ).fetchone()[0]
    assert collapsed == 1

    triage = triage_propositions(
        db_path=db_path,
        enriched_dir=enriched_dir,
        review_dir=review_dir,
        triage_run_id="RUN-DUP",
    )
    assert triage.triage_records_written == 11

    triage_rows = [json.loads(line) for line in Path(triage.output_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    prop_001 = next(row for row in triage_rows if row["proposition_id"] == "PROP-001")
    assert prop_001["evidence_coverage"]["claim_count"] == 2
    assert prop_001["observation_metrics"]["unique_observation_count"] == 2
    refs = prop_001["source_provenance"]["claim_refs"]
    assert len(refs) == 2
    assert len({item["source_file"] for item in refs}) == 2


def test_consensus_weighting_same_model_vs_cross_model_and_conflict(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "synthesis.sqlite3"
    enriched_dir = tmp_path / "enriched"
    review_dir = tmp_path / "review"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)
        conn.executemany(
            """
            INSERT INTO propositions (
              proposition_id, question, target, convergence, convergence_detail,
              resolved_type, caveat, exploration_axes, priority_hint, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("P-SAME", "same", "same", "convergent", "same", "supported", None, "[]", "medium", "2026-03-27T00:00:00+00:00"),
                ("P-CROSS", "cross", "cross", "convergent", "cross", "supported", None, "[]", "medium", "2026-03-27T00:00:00+00:00"),
                ("P-CONFLICT", "conflict", "conflict", "partial", "conflict", "supported", None, "[]", "high", "2026-03-27T00:00:00+00:00"),
            ],
        )
        conn.commit()

    _write_json(
        enriched_dir / "claude.json",
        {
            "meta": {"source_ai": "claude"},
            "claims": [
                {
                    "claim_id": "C-SAME-1",
                    "source_file": "claude/20260327.json",
                    "proposition_ids": ["P-SAME"],
                    "claim_type": "supported",
                    "claim": "same model repeat 1",
                    "evidence_files": ["same.csv"],
                    "evidence_refs": [],
                    "metrics_used": [],
                    "limitation_or_counterpoint": "limited period",
                },
                {
                    "claim_id": "C-SAME-2",
                    "source_file": "claude/20260328.json",
                    "proposition_ids": ["P-SAME"],
                    "claim_type": "supported",
                    "claim": "same model repeat 2",
                    "evidence_files": ["same.csv"],
                    "evidence_refs": [],
                    "metrics_used": [],
                    "limitation_or_counterpoint": "limited period",
                },
                {
                    "claim_id": "C-SAME-3",
                    "source_file": "claude/20260329.json",
                    "proposition_ids": ["P-SAME"],
                    "claim_type": "supported",
                    "claim": "same model repeat 3",
                    "evidence_files": ["same.csv"],
                    "evidence_refs": [],
                    "metrics_used": [],
                    "limitation_or_counterpoint": "limited period",
                },
                {
                    "claim_id": "C-CROSS-CLAUDE",
                    "source_file": "claude/20260327.json",
                    "proposition_ids": ["P-CROSS"],
                    "claim_type": "supported",
                    "claim": "cross model support from claude",
                    "evidence_files": ["cross_a.csv"],
                    "evidence_refs": ["A1"],
                    "metrics_used": [{"metric": "auc", "file": "cross_a.csv", "value": 1.1, "context": {"series_name": "daily", "condition": "base"}}],
                    "limitation_or_counterpoint": "limited period",
                },
                {
                    "claim_id": "C-CONFLICT-CLAUDE",
                    "source_file": "claude/20260327.json",
                    "proposition_ids": ["P-CONFLICT"],
                    "claim_type": "supported",
                    "claim": "conflict support side",
                    "evidence_files": ["conflict_pos.csv"],
                    "evidence_refs": ["P1"],
                    "metrics_used": [{"metric": "auc", "file": "conflict_pos.csv", "value": 1.1, "context": {"series_name": "daily", "condition": "base"}}],
                    "limitation_or_counterpoint": "limited period",
                },
            ],
        },
    )
    _write_json(
        enriched_dir / "gpt.json",
        {
            "meta": {"source_ai": "gpt"},
            "claims": [
                {
                    "claim_id": "C-CROSS-GPT",
                    "source_file": "gpt/20260327.json",
                    "proposition_ids": ["P-CROSS"],
                    "claim_type": "supported",
                    "claim": "cross model support from gpt",
                    "evidence_files": ["cross_b.csv"],
                    "evidence_refs": ["B1"],
                    "metrics_used": [{"metric": "auc", "file": "cross_b.csv", "value": 1.2, "context": {"series_name": "daily", "condition": "base"}}],
                    "limitation_or_counterpoint": "window mismatch",
                },
                {
                    "claim_id": "C-CONFLICT-GPT",
                    "source_file": "gpt/20260327.json",
                    "proposition_ids": ["P-CONFLICT"],
                    "claim_type": "negative",
                    "claim": "conflict negative side",
                    "evidence_files": ["conflict_neg.csv"],
                    "evidence_refs": ["N1"],
                    "metrics_used": [{"metric": "auc", "file": "conflict_neg.csv", "value": -0.8, "context": {"series_name": "daily", "condition": "base"}}],
                    "limitation_or_counterpoint": "method mismatch",
                },
            ],
        },
    )

    triage = triage_propositions(
        db_path=db_path,
        enriched_dir=enriched_dir,
        review_dir=review_dir,
        triage_run_id="RUN-CONSENSUS",
    )
    rows = [json.loads(line) for line in Path(triage.output_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["proposition_id"]: row for row in rows}
    same = by_id["P-SAME"]
    cross = by_id["P-CROSS"]
    conflict = by_id["P-CONFLICT"]

    assert same["consensus_class"] == "same_model_repeated"
    assert same["observation_metrics"]["unique_model_count"] == 1
    assert same["observation_metrics"]["same_model_repeat_count_max"] >= 2
    assert cross["consensus_class"] == "cross_model_supported"
    assert cross["observation_metrics"]["unique_model_count"] == 2
    assert cross["support_strength"] > same["support_strength"]
    assert conflict["consensus_class"] == "cross_model_conflicting"
    assert conflict["observation_metrics"]["has_cross_model_conflict"] is True
    assert conflict["review_status_suggested"] == "human_review_required"
    assert conflict["decision_risk"] > cross["decision_risk"]


def test_set_proposition_review_status_respects_locking_and_force(tmp_path: Path) -> None:
    db_path, enriched_dir, review_dir = _seed_propositions(tmp_path)
    triage_propositions(db_path=db_path, enriched_dir=enriched_dir, review_dir=review_dir)

    first = set_proposition_review_status(
        db_path=db_path,
        proposition_ids=["PROP-001"],
        review_status="triaged",
    )
    assert first.updated_records == 1
    assert first.history_records_written == 1

    second = set_proposition_review_status(
        db_path=db_path,
        proposition_ids=["PROP-001"],
        review_status="human_confirmed",
    )
    assert second.updated_records == 1

    locked = set_proposition_review_status(
        db_path=db_path,
        proposition_ids=["PROP-001"],
        review_status="triaged",
    )
    assert locked.updated_records == 0
    assert locked.skipped_locked_records == 1

    forced = set_proposition_review_status(
        db_path=db_path,
        proposition_ids=["PROP-001"],
        review_status="on_hold",
        allow_human_overwrite=True,
        force=True,
        reason="manual reopen",
    )
    assert forced.updated_records == 1
    assert forced.history_records_written == 1
    rendered = render_proposition_review_update_result(forced)
    assert "requested_status=on_hold" in rendered

    with sqlite3.connect(db_path) as conn:
        status = conn.execute(
            "SELECT current_status FROM proposition_review_state WHERE proposition_id = 'PROP-001'"
        ).fetchone()[0]
        history_count = conn.execute(
            "SELECT COUNT(*) FROM proposition_review_history WHERE proposition_id = 'PROP-001'"
        ).fetchone()[0]
    assert status == "on_hold"
    assert history_count == 3


def test_set_proposition_review_status_blocks_invalid_transition_without_force(tmp_path: Path) -> None:
    db_path, enriched_dir, review_dir = _seed_propositions(tmp_path)
    triage_propositions(db_path=db_path, enriched_dir=enriched_dir, review_dir=review_dir)

    blocked = set_proposition_review_status(
        db_path=db_path,
        proposition_ids=["PROP-002"],
        review_status="human_confirmed",
    )
    assert blocked.updated_records == 0
    assert blocked.skipped_transition_records == 1

    forced = set_proposition_review_status(
        db_path=db_path,
        proposition_ids=["PROP-002"],
        review_status="human_confirmed",
        force=True,
    )
    assert forced.updated_records == 1
