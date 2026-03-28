from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from arena.synthesis import ingest_validation as iv
from arena.synthesis.validator import LoadedRecord, ValidationIssue


def _valid_record() -> dict[str, object]:
    return {
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


def test_extract_source_date_and_infer_model_from_path(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    valid_path = raw_root / "claude" / "20260322-2.json"
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.write_text("[]", encoding="utf-8")

    assert iv.extract_source_date(Path("20260322.json")) == "2026-03-22"
    assert iv.extract_source_date(Path("20260322-0.json")) is None
    assert iv.extract_source_date(Path("bad_name.json")) is None
    assert iv.infer_model_from_path(valid_path, raw_root=raw_root) == "claude"

    outside = tmp_path / "outside" / "20260322.json"
    with pytest.raises(ValueError, match="outside raw root"):
        iv.infer_model_from_path(outside, raw_root=raw_root)

    with pytest.raises(ValueError, match="expected raw/<model>/<file>"):
        iv.infer_model_from_path(raw_root / "20260322.json", raw_root=raw_root)

    with pytest.raises(ValueError, match="unknown model directory"):
        iv.infer_model_from_path(raw_root / "unknown" / "20260322.json", raw_root=raw_root)


def test_load_and_validate_success_and_failure(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.json"
    invalid_path = tmp_path / "invalid.json"
    valid_path.write_text(json.dumps([_valid_record()], ensure_ascii=False), encoding="utf-8")
    invalid_path.write_text(json.dumps([{"claim": "missing required keys"}], ensure_ascii=False), encoding="utf-8")

    records, warning_count = iv.load_and_validate(valid_path)
    assert len(records) == 1
    assert warning_count == 0

    with pytest.raises(ValueError, match="validation failed for"):
        iv.load_and_validate(invalid_path)


def test_normalize_metric_helpers_cover_variants() -> None:
    actions: Counter[str] = Counter()

    assert iv._normalize_metric_context("bad", actions=actions) == {"series_name": None, "condition": None}
    normalized_context = iv._normalize_metric_context({"series_name": " daily ", "condition": " "}, actions=actions)
    assert normalized_context == {"series_name": "daily", "condition": None}
    assert actions["metrics_used_context_defaulted"] == 1
    assert actions["metrics_used_context_normalized"] == 1

    assert iv._normalize_metric_value(True, actions=actions) == "true"
    assert iv._normalize_metric_value(1.2, actions=actions) == 1.2
    assert iv._normalize_metric_value({"k": 1}, actions=actions) == '{"k": 1}'
    assert actions["metrics_used_value_bool_to_string"] == 1
    assert actions["metrics_used_value_other_to_string"] == 1

    assert iv._normalize_metrics_used(None, actions=actions) == []
    assert iv._normalize_metrics_used("", actions=actions) == []
    assert iv._normalize_metrics_used("single_metric", actions=actions)[0]["metric"] == "single_metric"
    split_metrics = iv._normalize_metrics_used("m1, m2", actions=actions)
    assert [item["metric"] for item in split_metrics] == ["m1", "m2"]
    assert iv._normalize_metrics_used({"metric": "auc"}, actions=actions)[0]["metric"] == "auc"
    assert iv._normalize_metrics_used(42, actions=actions) == []

    mixed = iv._normalize_metrics_used(
        [
            " metric_a ",
            " ",
            {"file": " m.csv ", "metric": " auc ", "value": True, "context": {"series_name": " s ", "condition": " "}},
            {"file": "x.csv", "metric": "loss", "value": {"x": 1}, "context": "unknown"},
            None,
            7,
        ],
        actions=actions,
    )
    assert mixed[0]["metric"] == "metric_a"
    assert mixed[1]["value"] == "true"
    assert mixed[2]["value"] == '{"x": 1}'
    assert mixed[3]["metric"] == "7"
    assert actions["metrics_used_blank_item_dropped"] >= 1
    assert actions["metrics_used_null_item_dropped"] >= 1
    assert actions["metrics_used_other_item_to_object"] >= 1


def test_shape_and_semantic_repair_record_normalize_fields() -> None:
    shape_actions: Counter[str] = Counter()
    shape_warnings: list[str] = []
    shaped = iv._shape_repair_record(
        {
            "claim": "c",
            "evidence_files": " metrics.csv ",
            "evidence_refs": " tbl:1 ",
            "required_files": {
                "file_name": " raw.csv ",
                "priority": "Priority A",
                "reason": " why ",
                "required_for": " eval ",
            },
            "basis_summary": " ",
            "limitation_or_counterpoint": ["  first ", "", "second"],
            "next_data_needed": ["  need-a ", "need-b "],
            "exploration_axes": " axis_a ",
            "fixed_conditions": " ",
            "variable_conditions": " v1 ",
        },
        actions=shape_actions,
        warnings=shape_warnings,
    )
    assert shaped["required_files"] is not None
    assert shaped["evidence_files"] == ["metrics.csv"]
    assert shaped["evidence_refs"] == ["tbl:1"]
    assert shaped["basis_summary"] is None
    assert shaped["limitation_or_counterpoint"] == "first\nsecond"
    assert shaped["next_data_needed"] == "need-a\nneed-b"
    assert shaped["exploration_axes"] == ["axis_a"]
    assert shaped["fixed_conditions"] is None
    assert shaped["variable_conditions"] == ["v1"]

    warning_actions: Counter[str] = Counter()
    warning_messages: list[str] = []
    warned = iv._shape_repair_record({"claim": "c", "required_files": "manual.csv"}, actions=warning_actions, warnings=warning_messages)
    assert warned["required_files"] == "manual.csv"
    assert warning_messages == ["required_files scalar string was kept for validator review."]

    semantic_actions: Counter[str] = Counter()
    semantic = iv._semantic_repair_record(
        {
            "metrics_used": "auc, loss",
            "evidence_level": "High",
            "priority_hint": " Priority A ",
            "created_at": "2026/03/27",
            "evidence_refs": " ref-1 ",
            "raw_text": [" line1 ", "line2 "],
            "required_files": [
                {"file_name": " raw.csv ", "priority": "priority b", "reason": " because ", "required_for": " eval "}
            ],
            "exploration_axes": [" axis ", ""],
            "fixed_conditions": [" cond-a ", 100],
            "variable_conditions": ["", " var-b "],
        },
        actions=semantic_actions,
    )
    assert [item["metric"] for item in semantic["metrics_used"]] == ["auc", "loss"]
    assert semantic["evidence_level"] == "inferred"
    assert semantic["priority_hint"] == "A"
    assert semantic["created_at"] == "2026-03-27T00:00:00"
    assert semantic["evidence_refs"] == ["ref-1"]
    assert semantic["raw_text"] == "line1\nline2"
    assert semantic["required_files"] == [
        {"file_name": "raw.csv", "priority": "B", "reason": "because", "required_for": "eval"}
    ]
    assert semantic["exploration_axes"] == ["axis"]
    assert semantic["fixed_conditions"] == ["cond-a"]
    assert semantic["variable_conditions"] == ["var-b"]
    assert semantic_actions["priority_hint_normalized"] == 1
    assert semantic_actions["created_at_normalized"] == 1
    assert semantic_actions["evidence_refs_normalized"] == 1
    assert semantic_actions["raw_text_normalized"] == 1
    assert semantic_actions["required_files_normalized"] == 1


def test_record_label_and_downgrade_semantic_errors() -> None:
    record_with_id = LoadedRecord(
        record_no=1,
        payload={"id": "CLAIM-001", "metrics_used": "a,b", "evidence_level": "strong"},
    )
    record_with_claim = LoadedRecord(record_no=2, payload={"claim_id": "C-2"})
    record_without_label = LoadedRecord(record_no=3, payload={})

    assert iv._record_label(record_with_id) == "CLAIM-001"
    assert iv._record_label(record_with_claim) == "C-2"
    assert iv._record_label(record_without_label) == "3"
    assert iv._record_label(None, fallback_record_no=9) == "9"
    assert iv._record_label(None) == "unknown"
    assert iv._preview_value("x" * 200, max_len=12) == "'xxxxxxxx..."

    assert iv._is_semantic_validation_error(
        ValidationIssue(level="ERROR", message="bad", record_no=1, field="evidence_level")
    )
    assert iv._is_semantic_validation_error(
        ValidationIssue(level="ERROR", message="must be list", record_no=1, field="metrics_used[0].metric")
    )
    assert not iv._is_semantic_validation_error(
        ValidationIssue(level="ERROR", message="priority keyword reserved", record_no=1, field="metrics_used")
    )
    assert not iv._is_semantic_validation_error(
        ValidationIssue(level="WARNING", message="warn", record_no=1, field="evidence_level")
    )

    issues = [
        ValidationIssue(level="ERROR", message="must be list", record_no=1, field="metrics_used"),
        ValidationIssue(level="ERROR", message="must be list", record_no=1, field="metrics_used[0].metric"),
        ValidationIssue(level="ERROR", message="invalid", record_no=2, field="evidence_level"),
        ValidationIssue(level="ERROR", message="priority keyword reserved", record_no=1, field="metrics_used"),
        ValidationIssue(level="WARNING", message="warn", record_no=3, field="claim"),
    ]
    kept, warnings = iv.downgrade_semantic_errors_to_warnings(
        records=[record_with_id],
        issues=issues,
    )
    assert len(warnings) == 2
    assert "record CLAIM-001 has metrics_used='a,b'" in warnings[0]
    assert "record 2 has evidence_level=None" in warnings[1]
    assert any(issue.level == "WARNING" and issue.field == "claim" for issue in kept)
    assert any("priority keyword reserved" in issue.message for issue in kept)


def test_repair_records_reports_parse_root_item_and_success_cases(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.json"

    invalid = iv.repair_records(source_path, "{", repair_semantic=False)
    assert invalid.error_message is not None
    assert "invalid JSON" in invalid.error_message

    wrong_root = iv.repair_records(source_path, "123", repair_semantic=False)
    assert wrong_root.error_message is not None
    assert "must be array/object" in wrong_root.error_message

    bad_item = iv.repair_records(source_path, '[{"claim": "ok"}, 3]', repair_semantic=False)
    assert bad_item.error_message is not None
    assert "record 2 is not object" in bad_item.error_message

    one_object = iv.repair_records(
        source_path,
        '{"claim": "ok", "required_files": " "}',
        repair_semantic=False,
    )
    assert one_object.error_message is None
    assert one_object.record_count_before == 1
    assert one_object.record_count_after == 1
    assert "single_object_to_array:1" in one_object.syntax_repairs
    assert any(item.startswith("required_files_empty_to_null:") for item in one_object.shape_repairs)
