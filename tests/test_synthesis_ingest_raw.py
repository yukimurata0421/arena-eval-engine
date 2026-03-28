from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from arena.synthesis import db as synthesis_db
from arena.synthesis.ingest import discover_raw_candidates, ingest_raw_tree


def _valid_record(with_metadata: bool = False) -> dict[str, object]:
    record: dict[str, object] = {
        "claim": "AUC improved by 5%.",
        "claim_type": "supported",
        "basis_summary": "Observed in experiment A.",
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
    if with_metadata:
        record.update(
            {
                "required_files": [
                    {
                        "file_name": "raw_events.csv",
                        "priority": "Priority A",
                        "reason": "needs recomputation",
                        "required_for": "revalidation",
                    }
                ],
                "raw_text": "Original raw claim sentence.",
                "evidence_refs": ["tbl:auc_monthly", "fig:trend_01"],
                "priority_hint": "high",
                "created_at": "2026-03-22T01:23:45Z",
                "topic": "model_quality",
                "topic_confidence": "high",
                "topic_method": "rule",
                "topic_reason": "matched keywords: auc, gain",
                "baseline_label": "v1.2",
                "baseline_type": "offline_batch",
                "baseline_confidence": "medium",
                "baseline_method": "rule",
                "baseline_reason": "explicit baseline mention",
                "exploration_axes": ["time_window", "segment"],
                "fixed_conditions": ["test_split=v3"],
                "variable_conditions": ["threshold=0.6", "threshold=0.7"],
                "validity_scope": "Q1-2026 JP region",
                "review_status": "pending",
            }
        )
    return record


def _patch_db(monkeypatch, tmp_path: Path) -> Path:
    db_dir = tmp_path / "db"
    db_path = db_dir / "synthesis.sqlite3"
    monkeypatch.setattr(synthesis_db, "DB_DIR", db_dir)
    monkeypatch.setattr(synthesis_db, "DB_PATH", db_path)
    monkeypatch.setattr(synthesis_db, "legacy_db_paths_present", lambda: [])
    return db_path


def _repair_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "raw_original",
        tmp_path / "raw_repaired",
        tmp_path / "repair_logs",
    )


def test_discover_raw_candidates_checks_layout_model_and_filename(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "claude").mkdir(parents=True)
    (raw / "other").mkdir(parents=True)
    (raw / "claude" / "nested").mkdir(parents=True)

    (raw / "claude" / "20260322.json").write_text("[]", encoding="utf-8")
    (raw / "claude" / "20260322-1.json").write_text("[]", encoding="utf-8")
    (raw / "claude" / "badname.json").write_text("[]", encoding="utf-8")
    (raw / "claude" / "20260322-0.jsonl").write_text("[]", encoding="utf-8")
    (raw / "other" / "20260322.json").write_text("[]", encoding="utf-8")
    (raw / "claude" / "nested" / "20260322.json").write_text("[]", encoding="utf-8")

    candidates, warnings, scanned = discover_raw_candidates(raw_dir=raw)

    assert scanned == 6
    assert sorted(candidate.name for candidate in candidates) == ["20260322-1.json", "20260322.json"]
    assert len(warnings) == 4


def test_ingest_raw_tree_skips_duplicates_by_file_hash(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    raw = tmp_path / "raw"
    (raw / "claude").mkdir(parents=True)
    (raw / "gemini").mkdir(parents=True)

    rec = json.dumps([_valid_record()], ensure_ascii=False)
    (raw / "claude" / "20260322.json").write_text(rec, encoding="utf-8")
    (raw / "gemini" / "20260322.json").write_text(rec, encoding="utf-8")

    first = ingest_raw_tree(raw_dir=raw)
    assert first.ingested_files == 2
    assert first.ingested_records == 2
    assert first.skipped_duplicates == 0
    assert first.failed_records == 0

    second = ingest_raw_tree(raw_dir=raw)
    assert second.ingested_files == 0
    assert second.ingested_records == 0
    assert second.skipped_duplicates == 2
    assert second.failed_records == 0

    with sqlite3.connect(db_path) as conn:
        hypotheses_count = conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
        ingested_files_count = conn.execute("SELECT COUNT(*) FROM ingested_files").fetchone()[0]

    assert hypotheses_count == 2
    assert ingested_files_count == 2


def test_ingest_raw_tree_extracts_source_date_from_slot_filename(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    raw = tmp_path / "raw"
    (raw / "claude").mkdir(parents=True)

    rec = json.dumps([_valid_record()], ensure_ascii=False)
    (raw / "claude" / "20260322-10.json").write_text(rec, encoding="utf-8")

    report = ingest_raw_tree(raw_dir=raw)
    assert report.ingested_files == 1
    assert report.ingested_records == 1

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT source_date FROM ingested_files").fetchone()

    assert row is not None
    assert row[0] == "2026-03-22"


def test_ingest_raw_tree_persists_required_files_and_claim_metadata(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    raw = tmp_path / "raw"
    (raw / "claude").mkdir(parents=True)

    rec = json.dumps([_valid_record(with_metadata=True)], ensure_ascii=False)
    (raw / "claude" / "20260322.json").write_text(rec, encoding="utf-8")

    report = ingest_raw_tree(raw_dir=raw)
    assert report.ingested_files == 1
    assert report.ingested_records == 1
    assert report.failed_records == 0

    with sqlite3.connect(db_path) as conn:
        hypothesis = conn.execute(
            """
            SELECT
              id, raw_text, evidence_refs, priority_hint, created_at,
              topic, topic_confidence, topic_method, topic_reason,
              baseline_label, baseline_type, baseline_confidence, baseline_method, baseline_reason,
              exploration_axes_json, fixed_conditions_json, variable_conditions_json,
              validity_scope, review_status
            FROM hypotheses
            """
        ).fetchone()
        required = conn.execute(
            "SELECT file_name, priority, reason, required_for FROM claim_required_files"
        ).fetchone()

    assert hypothesis is not None
    assert hypothesis[1] == "Original raw claim sentence."
    assert json.loads(hypothesis[2]) == ["tbl:auc_monthly", "fig:trend_01"]
    assert hypothesis[3] == "high"
    assert hypothesis[4] == "2026-03-22T01:23:45+00:00"
    assert hypothesis[5] == "model_quality"
    assert hypothesis[6] == "high"
    assert hypothesis[7] == "rule"
    assert hypothesis[8] == "matched keywords: auc, gain"
    assert hypothesis[9] == "v1.2"
    assert hypothesis[10] == "offline_batch"
    assert hypothesis[11] == "medium"
    assert hypothesis[12] == "rule"
    assert hypothesis[13] == "explicit baseline mention"
    assert json.loads(hypothesis[14]) == ["time_window", "segment"]
    assert json.loads(hypothesis[15]) == ["test_split=v3"]
    assert json.loads(hypothesis[16]) == ["threshold=0.6", "threshold=0.7"]
    assert hypothesis[17] == "Q1-2026 JP region"
    assert hypothesis[18] == "pending"

    assert required is not None
    assert required[0] == "raw_events.csv"
    assert required[1] == "A"
    assert required[2] == "needs recomputation"
    assert required[3] == "revalidation"


def test_ingest_raw_tree_normalizes_raw_text_list(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    raw = tmp_path / "raw"
    (raw / "claude").mkdir(parents=True)

    record = _valid_record()
    record["raw_text"] = ["detected_change_date=2026-01-10", "confidence_pct=98.93"]
    rec = json.dumps([record], ensure_ascii=False)
    (raw / "claude" / "20260322.json").write_text(rec, encoding="utf-8")

    report = ingest_raw_tree(raw_dir=raw)
    assert report.ingested_files == 1
    assert report.failed_records == 0

    with sqlite3.connect(db_path) as conn:
        hypothesis = conn.execute("SELECT raw_text FROM hypotheses").fetchone()

    assert hypothesis is not None
    assert hypothesis[0] == "detected_change_date=2026-01-10\nconfidence_pct=98.93"


def test_ingest_raw_tree_is_backward_compatible_when_metadata_missing(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    raw = tmp_path / "raw"
    (raw / "claude").mkdir(parents=True)

    legacy = _valid_record()
    legacy["required_files"] = None
    rec = json.dumps([legacy], ensure_ascii=False)
    (raw / "claude" / "20260322.json").write_text(rec, encoding="utf-8")

    report = ingest_raw_tree(raw_dir=raw)
    assert report.ingested_files == 1
    assert report.ingested_records == 1
    assert report.failed_records == 0

    with sqlite3.connect(db_path) as conn:
        hypothesis = conn.execute(
            """
            SELECT
              raw_text, evidence_refs, priority_hint,
              topic, topic_confidence, topic_method, topic_reason,
              baseline_label, baseline_type, baseline_confidence, baseline_method, baseline_reason,
              exploration_axes_json, fixed_conditions_json, variable_conditions_json,
              validity_scope, review_status
            FROM hypotheses
            """
        ).fetchone()
        required_count = conn.execute("SELECT COUNT(*) FROM claim_required_files").fetchone()[0]

    assert hypothesis is not None
    assert hypothesis[0] is None
    assert hypothesis[1] is None
    assert hypothesis[2] is None
    assert hypothesis[3] is None
    assert hypothesis[4] is None
    assert hypothesis[5] is None
    assert hypothesis[6] is None
    assert hypothesis[7] is None
    assert hypothesis[8] is None
    assert hypothesis[9] is None
    assert hypothesis[10] is None
    assert hypothesis[11] is None
    assert hypothesis[12] is None
    assert hypothesis[13] is None
    assert hypothesis[14] is None
    assert hypothesis[15] is None
    assert hypothesis[16] is None
    assert required_count == 0


def test_ingest_raw_tree_uses_baseline_candidate_fallback(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    raw = tmp_path / "raw"
    (raw / "gpt").mkdir(parents=True)

    record = _valid_record()
    record.update(
        {
            "baseline_label": None,
            "baseline_type": None,
            "baseline_confidence": None,
            "baseline_reason": None,
            "baseline_candidate": {
                "label": "RTL-SDR Default",
                "type": "hardware",
                "confidence": "high",
                "reason": "phase evaluator baseline",
            },
        }
    )
    rec = json.dumps([record], ensure_ascii=False)
    (raw / "gpt" / "20260322.json").write_text(rec, encoding="utf-8")

    report = ingest_raw_tree(raw_dir=raw)
    assert report.ingested_files == 1
    assert report.failed_records == 0

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT baseline_label, baseline_type, baseline_confidence, baseline_reason FROM hypotheses"
        ).fetchone()

    assert row is not None
    assert row[0] == "RTL-SDR Default"
    assert row[1] == "hardware"
    assert row[2] == "high"
    assert row[3] == "phase evaluator baseline"


def test_ingest_raw_tree_accepts_null_exploration_metadata(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    raw = tmp_path / "raw"
    (raw / "claude").mkdir(parents=True)

    record = _valid_record()
    record.update(
        {
            "topic": None,
            "baseline_label": None,
            "baseline_type": None,
            "baseline_confidence": None,
            "exploration_axes": None,
            "fixed_conditions": None,
            "variable_conditions": None,
            "validity_scope": None,
            "review_status": None,
        }
    )
    rec = json.dumps([record], ensure_ascii=False)
    (raw / "claude" / "20260322.json").write_text(rec, encoding="utf-8")

    report = ingest_raw_tree(raw_dir=raw)
    assert report.ingested_files == 1
    assert report.failed_records == 0

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
              topic, baseline_label, baseline_type, baseline_confidence,
              exploration_axes_json, fixed_conditions_json, variable_conditions_json,
              validity_scope, review_status
            FROM hypotheses
            """
        ).fetchone()

    assert row is not None
    assert all(value is None for value in row)


def test_ingest_raw_tree_counts_failed_records(monkeypatch, tmp_path: Path) -> None:
    _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()

    raw = tmp_path / "raw"
    (raw / "claude").mkdir(parents=True)

    good = json.dumps([_valid_record()], ensure_ascii=False)
    bad = json.dumps([{"claim": "missing required keys"}], ensure_ascii=False)
    (raw / "claude" / "20260322.json").write_text(good, encoding="utf-8")
    (raw / "claude" / "20260323.json").write_text(bad, encoding="utf-8")

    report = ingest_raw_tree(raw_dir=raw)
    assert report.scanned_candidates == 2
    assert report.ingested_files == 1
    assert report.ingested_records == 1
    assert report.failed_records == 1
    assert any("failed:" in warning for warning in report.warnings)


def test_repair_syntax_fixes_broken_json(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    raw_original_dir, raw_repaired_dir, repair_log_dir = _repair_dirs(tmp_path)

    raw = tmp_path / "raw"
    (raw / "gpt").mkdir(parents=True)
    malformed = """
[
  {
    "claim": "AUC improved by 5%.",
    "claim_type": "supported",
    "basis_summary": "Observed in experiment A.",
    "evidence_files": ["metrics.csv"],
    "metrics_used": [
      {
        "file": "metrics.csv",
        "metric": "auc",
        "value": 0.91,
        "context": {"series_name": "auc_trend", "condition": "test_split"}
      }
    ],
    "evidence_level": "direct",
    "limitation_or_counterpoint": null,
    "next_data_needed": "Long-term drift analysis.",
    "raw_text": [
      ""line1"",
      ""line "quoted" value""
    ]
  }
]
""".strip()
    (raw / "gpt" / "20260322.json").write_text(malformed, encoding="utf-8")

    failed = ingest_raw_tree(raw_dir=raw, repair=False)
    assert failed.failed_records == 1
    assert failed.ingested_files == 0

    repaired = ingest_raw_tree(
        raw_dir=raw,
        repair=True,
        raw_original_dir=raw_original_dir,
        raw_repaired_dir=raw_repaired_dir,
        repair_log_dir=repair_log_dir,
    )
    assert repaired.failed_records == 0
    assert repaired.ingested_files == 1
    assert repaired.original_saved == 1
    assert repaired.repaired_saved == 1
    assert repaired.repairs_applied >= 1
    assert repaired.warnings_count >= 1
    assert any("repair applied:" in warning for warning in repaired.warnings)

    original_files = list((raw_original_dir / "gpt").glob("*.json"))
    repaired_files = list((raw_repaired_dir / "gpt").glob("*.json"))
    assert len(original_files) == 1
    assert len(repaired_files) == 1
    assert original_files[0].read_text(encoding="utf-8") == malformed
    assert "line \\\"quoted\\\" value" in repaired_files[0].read_text(encoding="utf-8")

    log_path = repair_log_dir / "ingest_repair.jsonl"
    assert log_path.exists()
    logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert logs
    assert logs[-1]["repair_requested"] is True
    assert logs[-1]["validation_result"] == "passed"
    assert logs[-1]["db_ingest_result"] == "ingested"
    assert logs[-1]["inserted_records"] == 1
    assert logs[-1]["source_file"].endswith("20260322.json")
    assert logs[-1]["repaired_sha256"] == hashlib.sha256(
        repaired_files[0].read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT raw_text FROM hypotheses").fetchone()

    assert row is not None
    assert row[0] == 'line1\nline "quoted" value'


def test_repair_shape_fills_missing_keys(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    raw_original_dir, raw_repaired_dir, repair_log_dir = _repair_dirs(tmp_path)

    raw = tmp_path / "raw"
    (raw / "gpt").mkdir(parents=True)

    malformed_shape = {
        "claim": "AUC improved by 5%.",
        "claim_type": "supported",
        "basis_summary": "Observed in experiment A.",
        "evidence_files": "metrics.csv",
        "metrics_used": {
            "file": "metrics.csv",
            "metric": "auc",
            "value": 0.91,
            "context": {"series_name": "auc_trend", "condition": "test_split"},
        },
        "evidence_level": "direct",
        "limitation_or_counterpoint": None,
        "next_data_needed": "Long-term drift analysis.",
        "evidence_refs": "tbl:auc_monthly",
    }
    (raw / "gpt" / "20260322.json").write_text(json.dumps(malformed_shape, ensure_ascii=False), encoding="utf-8")

    report = ingest_raw_tree(
        raw_dir=raw,
        repair=True,
        raw_original_dir=raw_original_dir,
        raw_repaired_dir=raw_repaired_dir,
        repair_log_dir=repair_log_dir,
    )
    assert report.failed_records == 0
    assert report.ingested_records == 1

    with sqlite3.connect(db_path) as conn:
        refs = conn.execute("SELECT evidence_refs FROM hypotheses").fetchone()[0]

    assert json.loads(refs) == ["tbl:auc_monthly"]
    repaired_files = list((raw_repaired_dir / "gpt").glob("*.json"))
    payload = json.loads(repaired_files[0].read_text(encoding="utf-8"))
    assert payload[0]["evidence_files"] == ["metrics.csv"]
    assert isinstance(payload[0]["metrics_used"], dict)
    assert payload[0]["required_files"] is None


def test_ingest_raw_tree_repair_handles_inner_quote_and_double_wrapped_raw_text(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    raw_original_dir, raw_repaired_dir, repair_log_dir = _repair_dirs(tmp_path)

    raw = tmp_path / "raw"
    (raw / "gpt").mkdir(parents=True)
    malformed = """
[
  {
    "claim": "AUC improved by 5%.",
    "claim_type": "supported",
    "basis_summary": "Observed in experiment A.",
    "evidence_files": ["metrics.csv"],
    "evidence_refs": [
      "phase_evaluator.section1.phase2_vs_phase0",
      "bayesian_phase_results_cuda."Airspy Mini vs RTL-SDR""
    ],
    "required_files": null,
    "metrics_used": ["auc_n_used"],
    "evidence_level": "strong",
    "limitation_or_counterpoint": null,
    "next_data_needed": null,
    "raw_text": ""line1"\\n"line2""
  }
]
""".strip()
    (raw / "gpt" / "20260326.json").write_text(malformed, encoding="utf-8")

    failed = ingest_raw_tree(raw_dir=raw, repair=False)
    assert failed.failed_records == 1
    assert failed.ingested_files == 0

    repaired = ingest_raw_tree(
        raw_dir=raw,
        repair=True,
        raw_original_dir=raw_original_dir,
        raw_repaired_dir=raw_repaired_dir,
        repair_log_dir=repair_log_dir,
    )
    assert repaired.failed_records == 0
    assert repaired.ingested_files == 1
    assert repaired.ingested_records == 1

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT raw_text, evidence_refs FROM hypotheses").fetchone()

    assert row is not None
    assert row[0] == "line1\nline2"
    assert json.loads(row[1]) == [
        "phase_evaluator.section1.phase2_vs_phase0",
        'bayesian_phase_results_cuda."Airspy Mini vs RTL-SDR"',
    ]


def test_repair_semantic_normalizes_evidence_level(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    raw_original_dir, raw_repaired_dir, repair_log_dir = _repair_dirs(tmp_path)

    raw = tmp_path / "raw"
    (raw / "gpt").mkdir(parents=True)

    malformed = [
        {
            "claim": "daily_auc is improved.",
            "claim_type": "supported",
            "basis_summary": "derived from summary table.",
            "evidence_files": ["daily_metric_table.csv"],
            "metrics_used": ["daily_auc", "p_value", "effect_size"],
            "evidence_level": "strong（p<0.001、effect_size>0.9）",
            "limitation_or_counterpoint": None,
            "next_data_needed": "long-term validation",
        },
        {
            "claim": "coverage increased.",
            "claim_type": "supported",
            "basis_summary": "derived from phase evaluator output.",
            "evidence_files": ["phase_evaluator_results.csv"],
            "metrics_used": "auc_n_used, minutes_covered (threshold 1380)",
            "evidence_level": "High",
            "limitation_or_counterpoint": None,
            "next_data_needed": None,
        },
        {
            "claim": "weekday impact remains small.",
            "claim_type": "negative",
            "basis_summary": "weekday test output.",
            "evidence_files": ["weekday_test.csv"],
            "metrics_used": None,
            "evidence_level": None,
            "limitation_or_counterpoint": "sample size is limited",
            "next_data_needed": "more days",
        },
    ]
    (raw / "gpt" / "20260323.json").write_text(json.dumps(malformed, ensure_ascii=False), encoding="utf-8")

    report = ingest_raw_tree(
        raw_dir=raw,
        repair=True,
        repair_semantic=True,
        raw_original_dir=raw_original_dir,
        raw_repaired_dir=raw_repaired_dir,
        repair_log_dir=repair_log_dir,
    )
    assert report.failed_records == 0
    assert report.ingested_files == 1
    assert report.ingested_records == 3

    repaired_files = list((raw_repaired_dir / "gpt").glob("*.json"))
    payload = json.loads(repaired_files[0].read_text(encoding="utf-8"))
    first_metric = payload[0]["metrics_used"][0]
    assert first_metric == {
        "file": None,
        "metric": "daily_auc",
        "value": None,
        "context": {"series_name": None, "condition": None},
    }
    assert payload[0]["evidence_level"] == "inferred"
    assert payload[1]["evidence_level"] == "inferred"
    assert len(payload[1]["metrics_used"]) == 2
    assert payload[2]["metrics_used"] == []
    assert payload[2]["evidence_level"] == "inferred"

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
    assert count == 3


def test_repair_semantic_disabled_emits_warning(monkeypatch, tmp_path: Path) -> None:
    db_path = _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    raw_original_dir, raw_repaired_dir, repair_log_dir = _repair_dirs(tmp_path)

    raw = tmp_path / "raw"
    (raw / "gpt").mkdir(parents=True)

    malformed = [
        {
            "id": "CLAIM-003",
            "claim": "coverage increased.",
            "claim_type": "supported",
            "basis_summary": "derived from phase evaluator output.",
            "evidence_files": ["phase_evaluator_results.csv"],
            "metrics_used": "auc_n_used, minutes_covered (threshold 1380)",
            "evidence_level": "strong",
            "limitation_or_counterpoint": None,
            "next_data_needed": None,
        }
    ]
    (raw / "gpt" / "20260324.json").write_text(json.dumps(malformed, ensure_ascii=False), encoding="utf-8")

    report = ingest_raw_tree(
        raw_dir=raw,
        repair=True,
        repair_semantic=False,
        raw_original_dir=raw_original_dir,
        raw_repaired_dir=raw_repaired_dir,
        repair_log_dir=repair_log_dir,
    )
    assert report.failed_records == 0
    assert report.ingested_files == 1
    assert report.ingested_records == 1
    assert any("record CLAIM-003 has metrics_used=" in warning for warning in report.warnings)
    assert any("record CLAIM-003 has evidence_level=" in warning for warning in report.warnings)
    assert any("Use --repair-semantic to normalize." in warning for warning in report.warnings)

    repaired_files = list((raw_repaired_dir / "gpt").glob("*.json"))
    payload = json.loads(repaired_files[0].read_text(encoding="utf-8"))
    assert payload[0]["metrics_used"] == "auc_n_used, minutes_covered (threshold 1380)"
    assert payload[0]["evidence_level"] == "strong"

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
    assert count == 1


def test_ingest_raw_tree_repair_normalizes_optional_text_lists(monkeypatch, tmp_path: Path) -> None:
    _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    raw_original_dir, raw_repaired_dir, repair_log_dir = _repair_dirs(tmp_path)

    raw = tmp_path / "raw"
    (raw / "claude").mkdir(parents=True)

    malformed = _valid_record()
    malformed["limitation_or_counterpoint"] = [
        "  hardware change and gain tuning are close in time ",
        "",
        "causal attribution needs care",
    ]
    malformed["next_data_needed"] = [" change log audit ", "phase-boundary sensitivity analysis"]
    (raw / "claude" / "20260324.json").write_text(json.dumps([malformed], ensure_ascii=False), encoding="utf-8")

    report = ingest_raw_tree(
        raw_dir=raw,
        repair=True,
        raw_original_dir=raw_original_dir,
        raw_repaired_dir=raw_repaired_dir,
        repair_log_dir=repair_log_dir,
    )
    assert report.failed_records == 0
    assert report.validation_failed_files == 0
    assert report.ingested_files == 1
    assert report.ingested_records == 1

    repaired_files = list((raw_repaired_dir / "claude").glob("*.json"))
    payload = json.loads(repaired_files[0].read_text(encoding="utf-8"))
    assert payload[0]["limitation_or_counterpoint"] == (
        "hardware change and gain tuning are close in time\ncausal attribution needs care"
    )
    assert payload[0]["next_data_needed"] == "change log audit\nphase-boundary sensitivity analysis"

    log_path = repair_log_dir / "ingest_repair.jsonl"
    logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(item.startswith("limitation_or_counterpoint_normalized:") for item in logs[-1]["shape_repairs_applied"])
    assert any(item.startswith("next_data_needed_normalized:") for item in logs[-1]["shape_repairs_applied"])
    assert "repair_layer" in logs[-1]


def test_ingest_raw_tree_repair_still_rejects_unsafe_data(monkeypatch, tmp_path: Path) -> None:
    _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    raw_original_dir, raw_repaired_dir, repair_log_dir = _repair_dirs(tmp_path)

    raw = tmp_path / "raw"
    (raw / "gpt").mkdir(parents=True)

    unsafe = _valid_record()
    unsafe["metrics_used"] = [{"file": "metrics.csv", "metric": "priority", "value": "A", "context": {"series_name": "s", "condition": "c"}}]
    (raw / "gpt" / "20260322.json").write_text(json.dumps([unsafe], ensure_ascii=False), encoding="utf-8")

    report = ingest_raw_tree(
        raw_dir=raw,
        repair=True,
        raw_original_dir=raw_original_dir,
        raw_repaired_dir=raw_repaired_dir,
        repair_log_dir=repair_log_dir,
    )
    assert report.failed_records == 1
    assert report.validation_failed_files == 1
    assert report.ingested_files == 0

    log_path = repair_log_dir / "ingest_repair.jsonl"
    logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert logs[-1]["validation_result"] == "failed"
    assert logs[-1]["db_ingest_result"] == "not_ingested"
    assert isinstance(logs[-1]["repaired_sha256"], str)
    assert len(logs[-1]["repaired_sha256"]) == 64


def test_ingest_raw_tree_repair_duplicate_keeps_duplicate_logic(monkeypatch, tmp_path: Path) -> None:
    _patch_db(monkeypatch, tmp_path)
    synthesis_db.init_db()
    raw_original_dir, raw_repaired_dir, repair_log_dir = _repair_dirs(tmp_path)

    raw = tmp_path / "raw"
    (raw / "gpt").mkdir(parents=True)
    (raw / "gpt" / "20260322.json").write_text(json.dumps([_valid_record()], ensure_ascii=False), encoding="utf-8")

    first = ingest_raw_tree(
        raw_dir=raw,
        repair=True,
        raw_original_dir=raw_original_dir,
        raw_repaired_dir=raw_repaired_dir,
        repair_log_dir=repair_log_dir,
    )
    second = ingest_raw_tree(
        raw_dir=raw,
        repair=True,
        raw_original_dir=raw_original_dir,
        raw_repaired_dir=raw_repaired_dir,
        repair_log_dir=repair_log_dir,
    )
    assert first.ingested_files == 1
    assert second.skipped_duplicates == 1
    assert second.failed_records == 0

    log_path = repair_log_dir / "ingest_repair.jsonl"
    logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert logs[-1]["db_ingest_result"] == "skipped_duplicate"
    assert isinstance(logs[-1]["repaired_sha256"], str)
    assert len(logs[-1]["repaired_sha256"]) == 64
