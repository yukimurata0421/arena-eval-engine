from __future__ import annotations

import json
from pathlib import Path

from arena.synthesis import validator


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
                "raw_text": "Original sentence.",
                "evidence_refs": ["tbl:1", "fig:2"],
                "priority_hint": "high",
                "created_at": "2026-03-22T09:00:00Z",
                "topic": "model_quality",
                "topic_confidence": "high",
                "topic_method": "rule",
                "topic_reason": "matched keywords",
                "baseline_label": "v1.2",
                "baseline_type": "offline_batch",
                "baseline_confidence": "medium",
                "baseline_method": "rule",
                "baseline_reason": "explicit baseline",
                "exploration_axes": ["time_window", "segment"],
                "fixed_conditions": ["test_split=v3"],
                "variable_conditions": ["threshold=0.6", "threshold=0.7"],
                "validity_scope": "Q1-2026 JP region",
                "review_status": "pending",
            }
        )
    return record


def test_validator_ok_for_valid_json_array(tmp_path: Path, capsys) -> None:
    path = tmp_path / "valid.json"
    path.write_text(json.dumps([_valid_record(with_metadata=True)], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "OK: validated 1 record(s)." in captured.out


def test_validator_reports_required_and_enum_and_metrics_errors(tmp_path: Path, capsys) -> None:
    record = _valid_record()
    record.pop("claim_type")
    record["metrics_used"] = "auc=0.91"
    record["evidence_level"] = "weak"

    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "ERROR [record 1] claim_type: required key is missing." in captured.out
    assert "ERROR [record 1] metrics_used: expected list[object], got str (string values are not allowed)." in captured.out
    assert "ERROR [record 1] evidence_level: invalid value 'weak'" in captured.out
    assert "FAILED:" in captured.out


def test_validator_jsonl_warning_does_not_fail(tmp_path: Path, capsys) -> None:
    rec1 = _valid_record()
    rec1["evidence_files"] = []
    rec2 = _valid_record()

    path = tmp_path / "sample.jsonl"
    lines = [json.dumps(rec1, ensure_ascii=False), json.dumps(rec2, ensure_ascii=False)]
    path.write_text("\n".join(lines), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "WARNING [record 1, line 1] evidence_files: list is empty." in captured.out
    assert "OK: validated 2 record(s)." in captured.out


def test_validator_rejects_non_array_json_root(tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad_root.json"
    path.write_text(json.dumps(_valid_record(), ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "top-level JSON must be an array of objects" in captured.out
    assert "FAILED:" in captured.out


def test_validator_rejects_priority_mixed_into_metrics_used(tmp_path: Path, capsys) -> None:
    record = _valid_record()
    record["metrics_used"] = [
        {
            "file": "metrics.csv",
            "metric": "priority",
            "value": "A",
            "context": {"series_name": "auc_trend", "condition": "test_split"},
        }
    ]

    path = tmp_path / "priority_in_metrics.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "priority must not be represented in metrics_used" in captured.out


def test_validator_rejects_priority_hint_mixed_into_metrics_used(tmp_path: Path, capsys) -> None:
    record = _valid_record()
    record["metrics_used"] = [
        {
            "file": "metrics.csv",
            "metric": "priority_hint",
            "value": "high",
            "context": {"series_name": "auc_trend", "condition": "test_split"},
        }
    ]

    path = tmp_path / "priority_hint_in_metrics.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "priority_hint must not be represented in metrics_used" in captured.out


def test_validator_rejects_required_files_missing_file_name_or_blank_priority(tmp_path: Path, capsys) -> None:
    record = _valid_record()
    record["required_files"] = [
        {"priority": "A", "reason": "x", "required_for": "y"},
        {"file_name": "raw.csv", "priority": "   ", "reason": None, "required_for": None},
    ]

    path = tmp_path / "required_files_invalid.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "required_files[0].file_name: required key is missing." in captured.out
    assert "required_files[1].priority: must not be empty when provided." in captured.out


def test_validator_rejects_invalid_evidence_refs_type(tmp_path: Path, capsys) -> None:
    record = _valid_record()
    record["evidence_refs"] = "ref:1"

    path = tmp_path / "invalid_refs.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "evidence_refs: expected list[str] or null, got str." in captured.out


def test_validator_warns_on_invalid_created_at_format(tmp_path: Path, capsys) -> None:
    record = _valid_record()
    record["created_at"] = "22-03-2026"

    path = tmp_path / "invalid_created_at.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "created_at: unrecognized datetime format" in captured.out


def test_validator_rejects_priority_like_claim_type(tmp_path: Path, capsys) -> None:
    record = _valid_record()
    record["claim_type"] = "Priority A"

    path = tmp_path / "claim_type_priority.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "claim_type: priority-like value is not allowed here" in captured.out


def test_validator_accepts_raw_text_as_list_with_warning(tmp_path: Path, capsys) -> None:
    record = _valid_record()
    record["raw_text"] = ["line1", "line2"]

    path = tmp_path / "raw_text_list.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "raw_text: list input will be normalized to a newline-joined string during ingest." in captured.out


def test_validator_rejects_raw_text_list_with_non_string_item(tmp_path: Path, capsys) -> None:
    record = _valid_record()
    record["raw_text"] = ["line1", 123]

    path = tmp_path / "raw_text_list_bad_item.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "raw_text[1]: expected str, got int." in captured.out


def test_validator_accepts_null_exploration_metadata(tmp_path: Path, capsys) -> None:
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

    path = tmp_path / "exploration_null.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "OK: validated 1 record(s)." in captured.out


def test_validator_rejects_invalid_exploration_metadata_types(tmp_path: Path, capsys) -> None:
    record = _valid_record()
    record["topic"] = ["not", "string"]
    record["exploration_axes"] = "axis-a"

    path = tmp_path / "exploration_invalid.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    rc = validator.main(["--path", str(path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "topic: expected str or null, got list." in captured.out
    assert "exploration_axes: expected list[str] or null, got str." in captured.out
