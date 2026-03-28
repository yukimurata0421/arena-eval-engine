from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from arena.synthesis.ingest_models import MODEL_CHOICES, SOURCE_STEM_RE, RepairPipelineOutcome
from arena.synthesis.normalization import (
    normalize_created_at,
    normalize_evidence_refs,
    normalize_optional_string_list,
    normalize_optional_text,
    normalize_priority_hint,
    normalize_raw_text,
    normalize_required_files,
)
from arena.synthesis.repair import repair_json_text
from arena.synthesis.validator import LoadedRecord, ValidationIssue, load_records, validate_records

_EVIDENCE_LEVEL_STRENGTH_ALIASES: dict[str, str] = {
    "strong": "inferred",
    "high": "inferred",
    "moderate": "inferred",
    "medium": "inferred",
    "weak": "inferred",
    "low": "inferred",
}


def extract_source_date(path: Path) -> str | None:
    matched = SOURCE_STEM_RE.fullmatch(path.stem)
    if matched is None:
        return None

    slot = matched.group("slot")
    if slot is not None and int(slot) < 1:
        return None

    date = matched.group("date")
    return f"{date[0:4]}-{date[4:6]}-{date[6:8]}"


def infer_model_from_path(path: Path, raw_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(raw_root.resolve())
    except ValueError as exc:
        raise ValueError(f"cannot infer model from path outside raw root ({raw_root}): {path}") from exc

    if len(rel.parts) < 2:
        raise ValueError(f"cannot infer model from path: {path} (expected raw/<model>/<file>)")

    model = rel.parts[0].lower()
    if model not in MODEL_CHOICES:
        raise ValueError(f"unknown model directory {rel.parts[0]!r} in path: {path}")
    return model


def render_validation_summary(path: Path, issues: Sequence[ValidationIssue]) -> str:
    errors = [issue for issue in issues if issue.level == "ERROR"]
    warnings = [issue for issue in issues if issue.level == "WARNING"]
    lines = [f"validation failed for {path}: {len(errors)} error(s), {len(warnings)} warning(s)."]
    for issue in errors:
        lines.append(f"- {issue.format()}")
    return "\n".join(lines)


def load_and_validate(path: Path) -> tuple[list[LoadedRecord], int]:
    records, load_issues = load_records(path=path, format_hint="auto")
    validation_issues = validate_records(records)
    issues = [*load_issues, *validation_issues]
    error_count = sum(1 for issue in issues if issue.level == "ERROR")
    warning_count = sum(1 for issue in issues if issue.level == "WARNING")
    if error_count > 0:
        raise ValueError(render_validation_summary(path, issues))
    return records, warning_count


def _make_empty_metric_context() -> dict[str, str | None]:
    return {"series_name": None, "condition": None}


def _sanitize_metric_text(value: Any) -> str | None:
    return normalize_optional_text(value)


def _normalize_metric_context(value: Any, actions: Counter[str]) -> dict[str, str | None]:
    if not isinstance(value, dict):
        actions["metrics_used_context_defaulted"] += 1
        return _make_empty_metric_context()

    series_name = _sanitize_metric_text(value.get("series_name"))
    condition = _sanitize_metric_text(value.get("condition"))
    normalized = {"series_name": series_name, "condition": condition}
    if normalized != value:
        actions["metrics_used_context_normalized"] += 1
    return normalized


def _normalize_metric_value(value: Any, actions: Counter[str]) -> int | float | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        actions["metrics_used_value_bool_to_string"] += 1
        return str(value).lower()
    if isinstance(value, int | float | str):
        return value

    actions["metrics_used_value_other_to_string"] += 1
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_metrics_used(value: Any, actions: Counter[str]) -> list[dict[str, Any]]:
    if value is None:
        actions["metrics_used_missing_to_empty_array"] += 1
        return []

    entries = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            actions["metrics_used_empty_string_to_empty_array"] += 1
            return []
        metrics = [item.strip() for item in text.split(",") if item.strip()]
        if not metrics:
            actions["metrics_used_empty_string_to_empty_array"] += 1
            return []
        if len(metrics) > 1:
            actions["metrics_used_string_split_to_array"] += 1
            entries = metrics
        else:
            actions["metrics_used_string_to_array"] += 1
            entries = metrics
    elif isinstance(value, dict):
        actions["metrics_used_object_to_array"] += 1
        entries = [value]
    elif not isinstance(value, list):
        actions["metrics_used_scalar_to_empty_array"] += 1
        return []

    normalized_items: list[dict[str, Any]] = []
    assert isinstance(entries, list)
    for item in entries:
        if isinstance(item, str):
            metric_name = _sanitize_metric_text(item)
            if metric_name is None:
                actions["metrics_used_blank_item_dropped"] += 1
                continue
            normalized_items.append(
                {
                    "file": None,
                    "metric": metric_name,
                    "value": None,
                    "context": _make_empty_metric_context(),
                }
            )
            actions["metrics_used_string_item_to_object"] += 1
            continue

        if isinstance(item, dict):
            normalized_item = {
                "file": _sanitize_metric_text(item.get("file")),
                "metric": _sanitize_metric_text(item.get("metric")),
                "value": _normalize_metric_value(item.get("value"), actions=actions),
                "context": _normalize_metric_context(item.get("context"), actions=actions),
            }
            if normalized_item != item:
                actions["metrics_used_object_item_normalized"] += 1
            normalized_items.append(normalized_item)
            continue

        if item is None:
            actions["metrics_used_null_item_dropped"] += 1
            continue

        normalized_items.append(
            {
                "file": None,
                "metric": str(item),
                "value": None,
                "context": _make_empty_metric_context(),
            }
        )
        actions["metrics_used_other_item_to_object"] += 1

    return normalized_items


def _normalize_evidence_level(value: Any, actions: Counter[str]) -> str:
    if value is None:
        actions["evidence_level_missing_to_inferred"] += 1
        return "inferred"

    if not isinstance(value, str):
        actions["evidence_level_non_string_to_inferred"] += 1
        return "inferred"

    text = value.strip()
    if not text:
        actions["evidence_level_blank_to_inferred"] += 1
        return "inferred"

    lowered = text.casefold()
    if lowered in {"direct", "inferred"}:
        return lowered

    for alias, normalized in _EVIDENCE_LEVEL_STRENGTH_ALIASES.items():
        if lowered == alias or lowered.startswith(f"{alias}(") or lowered.startswith(f"{alias}（"):
            actions["evidence_level_strength_to_inferred"] += 1
            return normalized

    if "direct" in lowered:
        actions["evidence_level_text_to_direct"] += 1
        return "direct"

    if "infer" in lowered or "indirect" in lowered:
        actions["evidence_level_text_to_inferred"] += 1
        return "inferred"

    actions["evidence_level_unknown_to_inferred"] += 1
    return "inferred"


def _shape_repair_record(record: dict[str, Any], actions: Counter[str], warnings: list[str]) -> dict[str, Any]:
    repaired = dict(record)

    if "required_files" not in repaired:
        repaired["required_files"] = None
        actions["required_files_missing_to_null"] += 1

    if "metrics_used" not in repaired:
        repaired["metrics_used"] = []
        actions["metrics_used_missing_to_empty_array"] += 1

    if "evidence_level" not in repaired:
        repaired["evidence_level"] = "inferred"
        actions["evidence_level_missing_to_inferred"] += 1

    evidence_files = repaired.get("evidence_files")
    if isinstance(evidence_files, str):
        text = evidence_files.strip()
        repaired["evidence_files"] = [text] if text else []
        actions["evidence_files_string_to_array"] += 1

    evidence_refs = repaired.get("evidence_refs")
    if isinstance(evidence_refs, str):
        text = evidence_refs.strip()
        repaired["evidence_refs"] = [text] if text else []
        actions["evidence_refs_string_to_array"] += 1

    required_files = repaired.get("required_files")
    if isinstance(required_files, dict):
        repaired["required_files"] = [required_files]
        actions["required_files_object_to_array"] += 1
    elif isinstance(required_files, str):
        if not required_files.strip():
            repaired["required_files"] = None
            actions["required_files_empty_to_null"] += 1
        else:
            warnings.append("required_files scalar string was kept for validator review.")

    for field in ("limitation_or_counterpoint", "next_data_needed"):
        if field in repaired:
            normalized_text = normalize_raw_text(repaired.get(field))
            if normalized_text != repaired.get(field):
                repaired[field] = normalized_text
                actions[f"{field}_normalized"] += 1

    nullable_text_fields = (
        "basis_summary",
        "limitation_or_counterpoint",
        "next_data_needed",
        "priority_hint",
        "created_at",
        "topic",
        "primary_topic",
        "topic_confidence",
        "topic_method",
        "topic_reason",
        "baseline_label",
        "baseline_type",
        "baseline_confidence",
        "baseline_method",
        "baseline_reason",
        "validity_scope",
        "review_status",
    )
    for field in nullable_text_fields:
        value = repaired.get(field)
        if isinstance(value, str) and not value.strip():
            repaired[field] = None
            actions[f"{field}_empty_to_null"] += 1

    for field in ("exploration_axes", "fixed_conditions", "variable_conditions"):
        value = repaired.get(field)
        if isinstance(value, str):
            if not value.strip():
                repaired[field] = None
                actions[f"{field}_empty_to_null"] += 1
            else:
                repaired[field] = [value.strip()]
                actions[f"{field}_string_to_array"] += 1

    return repaired


def _semantic_repair_record(record: dict[str, Any], actions: Counter[str]) -> dict[str, Any]:
    repaired = dict(record)

    normalized_metrics_used = _normalize_metrics_used(repaired.get("metrics_used"), actions=actions)
    if normalized_metrics_used != repaired.get("metrics_used"):
        repaired["metrics_used"] = normalized_metrics_used

    normalized_evidence_level = _normalize_evidence_level(repaired.get("evidence_level"), actions=actions)
    if normalized_evidence_level != repaired.get("evidence_level"):
        repaired["evidence_level"] = normalized_evidence_level

    if "priority_hint" in repaired:
        normalized_hint = normalize_priority_hint(repaired.get("priority_hint"))
        if normalized_hint != repaired.get("priority_hint"):
            repaired["priority_hint"] = normalized_hint
            actions["priority_hint_normalized"] += 1

    if "created_at" in repaired:
        normalized_created_at = normalize_created_at(repaired.get("created_at"))
        if normalized_created_at != repaired.get("created_at"):
            repaired["created_at"] = normalized_created_at
            actions["created_at_normalized"] += 1

    if "evidence_refs" in repaired:
        normalized_refs = normalize_evidence_refs(repaired.get("evidence_refs"))
        if normalized_refs != repaired.get("evidence_refs"):
            repaired["evidence_refs"] = normalized_refs
            actions["evidence_refs_normalized"] += 1

    if "raw_text" in repaired:
        normalized_raw_text = normalize_raw_text(repaired.get("raw_text"))
        if normalized_raw_text != repaired.get("raw_text"):
            repaired["raw_text"] = normalized_raw_text
            actions["raw_text_normalized"] += 1

    required_files = repaired.get("required_files")
    if isinstance(required_files, list):
        normalized_required_files = normalize_required_files(required_files)
        if normalized_required_files != required_files:
            repaired["required_files"] = normalized_required_files
            actions["required_files_normalized"] += 1

    for field in ("exploration_axes", "fixed_conditions", "variable_conditions"):
        value = repaired.get(field)
        if isinstance(value, list):
            normalized_items = normalize_optional_string_list(value)
            if normalized_items != value:
                repaired[field] = normalized_items
                actions[f"{field}_normalized"] += 1

    return repaired


def _summarize_actions(counter: Counter[str]) -> tuple[str, ...]:
    return tuple(f"{name}:{counter[name]}" for name in sorted(counter))


def _record_label(record: LoadedRecord | None, fallback_record_no: int | None = None) -> str:
    if record is not None:
        record_id = normalize_optional_text(record.payload.get("id"))
        if record_id is not None:
            return record_id
        claim_id = normalize_optional_text(record.payload.get("claim_id"))
        if claim_id is not None:
            return claim_id
        return str(record.record_no)
    if fallback_record_no is not None:
        return str(fallback_record_no)
    return "unknown"


def _preview_value(value: Any, max_len: int = 120) -> str:
    text = repr(value)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def _is_semantic_validation_error(issue: ValidationIssue) -> bool:
    if issue.level != "ERROR" or issue.field is None:
        return False
    if issue.field == "evidence_level":
        return True
    if issue.field == "metrics_used" or issue.field.startswith("metrics_used["):
        return "priority" not in issue.message.casefold()
    return False


def downgrade_semantic_errors_to_warnings(
    *,
    records: Sequence[LoadedRecord],
    issues: Sequence[ValidationIssue],
) -> tuple[list[ValidationIssue], list[str]]:
    record_by_no = {record.record_no: record for record in records}
    kept: list[ValidationIssue] = []
    warnings: list[str] = []
    seen: set[tuple[int | None, str]] = set()

    for issue in issues:
        if not _is_semantic_validation_error(issue):
            kept.append(issue)
            continue

        if issue.field is None:
            continue

        if issue.field.startswith("metrics_used"):
            kind = "metrics_used"
            expected = "list[object]"
        elif issue.field.startswith("evidence_level"):
            kind = "evidence_level"
            expected = "direct|inferred"
        else:
            kept.append(issue)
            continue

        key = (issue.record_no, kind)
        if key in seen:
            continue
        seen.add(key)

        record = record_by_no.get(issue.record_no or -1)
        label = _record_label(record, fallback_record_no=issue.record_no)
        value = record.payload.get(kind) if record is not None else None
        warnings.append(
            f"WARNING: record {label} has {kind}={_preview_value(value)} "
            f"(expected: {expected}). Use --repair-semantic to normalize."
        )

    return kept, warnings


def repair_records(source_path: Path, source_text: str, *, repair_semantic: bool) -> RepairPipelineOutcome:
    syntax_result = repair_json_text(source_text)
    syntax_repairs = tuple(syntax_result.applied_rules)

    try:
        parsed = json.loads(syntax_result.text)
    except json.JSONDecodeError as exc:
        return RepairPipelineOutcome(
            records=(),
            repaired_text=syntax_result.text,
            syntax_repairs=syntax_repairs,
            shape_repairs=(),
            semantic_repairs=(),
            warnings=(),
            record_count_before=0,
            record_count_after=0,
            error_message=(
                f"{source_path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}."
            ),
        )

    if isinstance(parsed, dict):
        parsed = [parsed]
        syntax_repairs = (*syntax_repairs, "single_object_to_array:1")

    if not isinstance(parsed, list):
        return RepairPipelineOutcome(
            records=(),
            repaired_text=syntax_result.text,
            syntax_repairs=syntax_repairs,
            shape_repairs=(),
            semantic_repairs=(),
            warnings=(),
            record_count_before=0,
            record_count_after=0,
            error_message=f"{source_path}: repaired payload root must be array/object, got {type(parsed).__name__}.",
        )

    shape_actions: Counter[str] = Counter()
    semantic_actions: Counter[str] = Counter()
    warnings: list[str] = []
    repaired_payloads: list[dict[str, Any]] = []

    for idx, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            return RepairPipelineOutcome(
                records=(),
                repaired_text=syntax_result.text,
                syntax_repairs=syntax_repairs,
                shape_repairs=_summarize_actions(shape_actions),
                semantic_repairs=_summarize_actions(semantic_actions),
                warnings=tuple(warnings),
                record_count_before=len(parsed),
                record_count_after=len(repaired_payloads),
                error_message=f"{source_path}: record {idx} is not object (got {type(item).__name__}).",
            )

        shaped = _shape_repair_record(item, actions=shape_actions, warnings=warnings)
        if repair_semantic:
            repaired_payloads.append(_semantic_repair_record(shaped, actions=semantic_actions))
        else:
            repaired_payloads.append(shaped)

    repaired_text = json.dumps(repaired_payloads, ensure_ascii=False, indent=2)
    loaded_records = tuple(
        LoadedRecord(record_no=i, payload=payload) for i, payload in enumerate(repaired_payloads, start=1)
    )

    return RepairPipelineOutcome(
        records=loaded_records,
        repaired_text=repaired_text,
        syntax_repairs=syntax_repairs,
        shape_repairs=_summarize_actions(shape_actions),
        semantic_repairs=_summarize_actions(semantic_actions),
        warnings=tuple(warnings),
        record_count_before=len(parsed),
        record_count_after=len(repaired_payloads),
        error_message=None,
    )
