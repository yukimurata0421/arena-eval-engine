from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from arena.synthesis.proposition_layer import build_proposition_layer
from arena.synthesis.proposition_review_queue_status import (
    _is_transition_allowed,
    _normalize_optional_review_status,
    _normalize_proposition_ids,
    _normalize_review_status,
    export_review_queue,
    set_proposition_review_status,
)
from arena.synthesis.proposition_review_triage import triage_propositions


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_review_context(tmp_path: Path) -> tuple[Path, Path]:
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
                "metrics_used": [
                    {
                        "file": "change_point_report.csv",
                        "metric": "auc_n_used",
                        "value": 1.2,
                        "context": {"series_name": "daily", "condition": "baseline"},
                    }
                ],
                "evidence_level": "direct",
                "limitation_or_counterpoint": "Out-of-window behavior is unverified.",
                "next_data_needed": "Revalidate with additional windows.",
                "exploration_axes": ["coverage_auc", "baseline"],
            }
        ],
    )
    build_proposition_layer(raw_dir=raw_dir, enriched_dir=enriched_dir, db_path=db_path)
    triage_propositions(
        db_path=db_path,
        enriched_dir=enriched_dir,
        review_dir=review_dir,
        triage_run_id="RUN-QUEUE",
    )
    return db_path, review_dir


def test_export_review_queue_csv_and_filter_locked_status(tmp_path: Path) -> None:
    db_path, review_dir = _seed_review_context(tmp_path)

    first = set_proposition_review_status(db_path=db_path, proposition_ids=["PROP-001"], review_status="triaged")
    assert first.updated_records == 1
    second = set_proposition_review_status(
        db_path=db_path,
        proposition_ids=["PROP-001"],
        review_status="human_confirmed",
    )
    assert second.updated_records == 1

    csv_report = export_review_queue(
        db_path=db_path,
        review_dir=review_dir,
        format_name="csv",
        include_final=False,
    )
    assert csv_report.format == "csv"
    csv_path = Path(csv_report.output_path)
    assert csv_path.exists()

    with csv_path.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.reader(fp))
    header, data_rows = rows[0], rows[1:]
    assert "recommended_action" in header
    assert all(row[1] != "PROP-001" for row in data_rows)

    included = export_review_queue(
        db_path=db_path,
        review_dir=review_dir,
        format_name="csv",
        include_final=True,
        limit=1,
    )
    with Path(included.output_path).open(encoding="utf-8", newline="") as fp:
        included_rows = list(csv.reader(fp))
    assert len(included_rows) == 2


def test_export_review_queue_validates_arguments(tmp_path: Path) -> None:
    db_path, review_dir = _seed_review_context(tmp_path)

    with pytest.raises(ValueError, match="format_name must be either"):
        export_review_queue(db_path=db_path, review_dir=review_dir, format_name="txt")
    with pytest.raises(ValueError, match="limit must be >= 1"):
        export_review_queue(db_path=db_path, review_dir=review_dir, format_name="jsonl", limit=0)


def test_queue_status_normalizers_and_transition_helpers() -> None:
    assert _normalize_review_status(" Triaged ", field_name="review_status") == "triaged"
    with pytest.raises(ValueError, match="must be one of"):
        _normalize_review_status(None, field_name="review_status")
    with pytest.raises(ValueError, match="must be one of"):
        _normalize_review_status("not-a-status", field_name="review_status")

    assert _normalize_optional_review_status(None, field_name="where_current_status") is None
    assert _normalize_optional_review_status(" ON_HOLD ", field_name="where_current_status") == "on_hold"

    assert _normalize_proposition_ids(None) == []
    assert _normalize_proposition_ids([" ", "PROP-1", "PROP-1", " PROP-2 "]) == ["PROP-1", "PROP-2"]

    assert _is_transition_allowed("triaged", "triaged")
    assert not _is_transition_allowed("pending", "human_confirmed")


def test_set_review_status_handles_missing_filters_limit_and_unchanged(tmp_path: Path) -> None:
    db_path, _ = _seed_review_context(tmp_path)

    with sqlite3.connect(db_path) as conn:
        triage_id = conn.execute(
            "SELECT id FROM proposition_triage_records WHERE proposition_id = 'PROP-001' ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]

    report = set_proposition_review_status(
        db_path=db_path,
        proposition_ids=["PROP-001", "PROP-MISSING"],
        triage_record_ids=[triage_id, 999999],
        review_status="triaged",
        limit=1,
        reason="  queue sweep  ",
    )
    assert report.updated_records == 1
    assert report.reason_used == "queue sweep"
    assert "triage_record:999999" in report.skipped_missing_ids
    assert "PROP-MISSING" in report.skipped_missing_ids

    unchanged = set_proposition_review_status(
        db_path=db_path,
        proposition_ids=["PROP-001"],
        review_status="triaged",
    )
    assert unchanged.unchanged_records == 1

    filtered = set_proposition_review_status(
        db_path=db_path,
        proposition_ids=["PROP-001"],
        review_status="human_review_required",
        where_current_status="on_hold",
    )
    assert filtered.matched_records == 0
    assert filtered.updated_records == 0

    with pytest.raises(ValueError, match="limit must be >= 1"):
        set_proposition_review_status(db_path=db_path, proposition_ids=["PROP-001"], review_status="triaged", limit=0)
    with pytest.raises(ValueError, match="Specify at least one selector"):
        set_proposition_review_status(db_path=db_path, review_status="triaged")

    only_missing_selector = set_proposition_review_status(
        db_path=db_path,
        triage_record_ids=[999998],
        review_status="triaged",
    )
    assert only_missing_selector.updated_records == 0
    assert only_missing_selector.matched_records == 0
    assert "triage_record:999998" in only_missing_selector.skipped_missing_ids
