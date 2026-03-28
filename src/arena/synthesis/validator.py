from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arena.synthesis.normalization import (
    is_valid_created_at,
    normalize_created_at,
    normalize_evidence_refs,
    normalize_optional_string_list,
    normalize_optional_text,
    normalize_priority,
    normalize_priority_hint,
    normalize_raw_text,
)

ALLOWED_CLAIM_TYPES = ("supported", "negative", "unknown", "future")
ALLOWED_EVIDENCE_LEVELS = ("direct", "inferred")
REQUIRED_RECORD_KEYS = (
    "claim",
    "claim_type",
    "basis_summary",
    "evidence_files",
    "metrics_used",
    "evidence_level",
    "limitation_or_counterpoint",
    "next_data_needed",
)
REQUIRED_METRIC_KEYS = ("file", "metric", "value", "context")
REQUIRED_CONTEXT_KEYS = ("series_name", "condition")
PRIORITY_VALUE_RE = re.compile(r"^(?:a|b|c|high|medium|low|priority\s+[abc])$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LoadedRecord:
    record_no: int
    payload: dict[str, Any]
    line_no: int | None = None


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    level: str
    message: str
    record_no: int | None = None
    field: str | None = None
    line_no: int | None = None

    def format(self) -> str:
        location: list[str] = []
        if self.record_no is not None:
            location.append(f"record {self.record_no}")
        if self.line_no is not None:
            location.append(f"line {self.line_no}")

        prefix = self.level
        if location:
            prefix += f" [{', '.join(location)}]"
        if self.field:
            return f"{prefix} {self.field}: {self.message}"
        return f"{prefix} {self.message}"


def _type_name(value: Any) -> str:
    return type(value).__name__


def _is_str_or_none(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _is_metric_value_type(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    return isinstance(value, int | float | str)


def _format_choices(choices: Sequence[str]) -> str:
    return ", ".join(choices)


def _detect_format(path: Path, format_hint: str) -> str:
    if format_hint != "auto":
        return format_hint
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return "jsonl"
    return "json"


def _contains_priority_signal(value: str) -> bool:
    lowered = " ".join(value.casefold().split())
    if "priority" in lowered:
        return True
    return PRIORITY_VALUE_RE.fullmatch(lowered) is not None


def _looks_like_priority_field(value: str) -> bool:
    lowered = value.casefold()
    return "priority" in lowered


def _load_json_records(text: str, path: Path) -> tuple[list[LoadedRecord], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    stripped = text.strip()
    if not stripped:
        issues.append(ValidationIssue(level="ERROR", message=f"{path}: file is empty."))
        return [], issues

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        issues.append(
            ValidationIssue(
                level="ERROR",
                message=f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}.",
            )
        )
        return [], issues

    if not isinstance(parsed, list):
        issues.append(
            ValidationIssue(
                level="ERROR",
                message=f"{path}: top-level JSON must be an array of objects, got {_type_name(parsed)}.",
            )
        )
        return [], issues

    records: list[LoadedRecord] = []
    for idx, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            issues.append(
                ValidationIssue(
                    level="ERROR",
                    record_no=idx,
                    field="$",
                    message=f"expected object, got {_type_name(item)}.",
                )
            )
            continue
        records.append(LoadedRecord(record_no=idx, payload=item))
    return records, issues


def _load_jsonl_records(text: str, path: Path) -> tuple[list[LoadedRecord], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    records: list[LoadedRecord] = []
    stripped = text.strip()
    if not stripped:
        issues.append(ValidationIssue(level="ERROR", message=f"{path}: file is empty."))
        return [], issues

    record_no = 0
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    level="ERROR",
                    line_no=line_no,
                    message=f"{path}: invalid JSONL record at line {line_no}: {exc.msg}.",
                )
            )
            continue

        if not isinstance(item, dict):
            issues.append(
                ValidationIssue(
                    level="ERROR",
                    line_no=line_no,
                    message=f"{path}: JSONL line must be an object, got {_type_name(item)}.",
                )
            )
            continue

        record_no += 1
        records.append(LoadedRecord(record_no=record_no, payload=item, line_no=line_no))

    return records, issues


def load_records(path: Path, format_hint: str = "auto") -> tuple[list[LoadedRecord], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    if not path.exists():
        issues.append(ValidationIssue(level="ERROR", message=f"file does not exist: {path}"))
        return [], issues
    if not path.is_file():
        issues.append(ValidationIssue(level="ERROR", message=f"path is not a file: {path}"))
        return [], issues

    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        issues.append(ValidationIssue(level="ERROR", message=f"failed to read {path}: {exc}"))
        return [], issues

    input_format = _detect_format(path=path, format_hint=format_hint)
    if input_format == "jsonl":
        return _load_jsonl_records(text=text, path=path)
    return _load_json_records(text=text, path=path)


def _validate_optional_string(
    value: Any,
    field: str,
    record_no: int,
    line_no: int | None,
    issues: list[ValidationIssue],
) -> None:
    if _is_str_or_none(value):
        return
    issues.append(
        ValidationIssue(
            level="ERROR",
            record_no=record_no,
            line_no=line_no,
            field=field,
            message=f"expected str or null, got {_type_name(value)}.",
        )
    )


def _validate_optional_string_list(
    value: Any,
    field: str,
    record_no: int,
    line_no: int | None,
    issues: list[ValidationIssue],
) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        issues.append(
            ValidationIssue(
                level="ERROR",
                record_no=record_no,
                line_no=line_no,
                field=field,
                message=f"expected list[str] or null, got {_type_name(value)}.",
            )
        )
        return

    for idx, item in enumerate(value):
        if not isinstance(item, str):
            issues.append(
                ValidationIssue(
                    level="ERROR",
                    record_no=record_no,
                    line_no=line_no,
                    field=f"{field}[{idx}]",
                    message=f"expected str, got {_type_name(item)}.",
                )
            )
        elif normalize_optional_text(item) is None:
            issues.append(
                ValidationIssue(
                    level="WARNING",
                    record_no=record_no,
                    line_no=line_no,
                    field=f"{field}[{idx}]",
                    message="blank string is ignored.",
                )
            )

    normalized = normalize_optional_string_list(value)
    if normalized is None and value:
        issues.append(
            ValidationIssue(
                level="WARNING",
                record_no=record_no,
                line_no=line_no,
                field=field,
                message="all entries are blank after normalization.",
            )
        )


def validate_record(record: Mapping[str, Any], record_no: int, line_no: int | None = None) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add_error(field: str, message: str) -> None:
        issues.append(
            ValidationIssue(
                level="ERROR",
                record_no=record_no,
                line_no=line_no,
                field=field,
                message=message,
            )
        )

    def add_warning(field: str, message: str) -> None:
        issues.append(
            ValidationIssue(
                level="WARNING",
                record_no=record_no,
                line_no=line_no,
                field=field,
                message=message,
            )
        )

    for key in REQUIRED_RECORD_KEYS:
        if key not in record:
            add_error(key, "required key is missing.")

    claim = record.get("claim")
    if "claim" in record:
        if not isinstance(claim, str):
            add_error("claim", f"expected str, got {_type_name(claim)}.")
        elif not claim.strip():
            add_error("claim", "must not be empty.")

    claim_type = record.get("claim_type")
    if "claim_type" in record:
        if not isinstance(claim_type, str):
            add_error("claim_type", f"expected one of ({_format_choices(ALLOWED_CLAIM_TYPES)}), got {_type_name(claim_type)}.")
        elif claim_type not in ALLOWED_CLAIM_TYPES:
            if _contains_priority_signal(claim_type):
                add_error("claim_type", "priority-like value is not allowed here; use required_files.priority or priority_hint.")
            else:
                add_error(
                    "claim_type",
                    f"invalid value {claim_type!r}; expected one of ({_format_choices(ALLOWED_CLAIM_TYPES)}).",
                )

    if "basis_summary" in record:
        _validate_optional_string(
            value=record["basis_summary"],
            field="basis_summary",
            record_no=record_no,
            line_no=line_no,
            issues=issues,
        )

    evidence_files = record.get("evidence_files")
    normalized_evidence_files: list[str] = []
    if "evidence_files" in record:
        if not isinstance(evidence_files, list):
            add_error("evidence_files", f"expected list[str], got {_type_name(evidence_files)}.")
        else:
            if not evidence_files:
                add_warning("evidence_files", "list is empty.")
            for idx, item in enumerate(evidence_files):
                if not isinstance(item, str):
                    add_error(f"evidence_files[{idx}]", f"expected str, got {_type_name(item)}.")
                    continue
                stripped = item.strip()
                if not stripped:
                    add_warning(f"evidence_files[{idx}]", "blank string is ignored.")
                    continue
                normalized_evidence_files.append(stripped)

    metrics_used = record.get("metrics_used")
    if "metrics_used" in record:
        if isinstance(metrics_used, str):
            add_error("metrics_used", "expected list[object], got str (string values are not allowed).")
        elif not isinstance(metrics_used, list):
            add_error("metrics_used", f"expected list[object], got {_type_name(metrics_used)}.")
        else:
            for idx, metric_item in enumerate(metrics_used):
                field_prefix = f"metrics_used[{idx}]"
                if not isinstance(metric_item, dict):
                    add_error(field_prefix, f"expected object, got {_type_name(metric_item)}.")
                    continue

                for key in REQUIRED_METRIC_KEYS:
                    if key not in metric_item:
                        add_error(f"{field_prefix}.{key}", "required key is missing.")

                metric_name = metric_item.get("metric")
                if "file" in metric_item and not _is_str_or_none(metric_item["file"]):
                    add_error(f"{field_prefix}.file", f"expected str or null, got {_type_name(metric_item['file'])}.")

                if "metric" in metric_item and not _is_str_or_none(metric_name):
                    add_error(f"{field_prefix}.metric", f"expected str or null, got {_type_name(metric_name)}.")
                elif isinstance(metric_name, str) and _looks_like_priority_field(metric_name):
                    if "priority_hint" in metric_name.casefold():
                        add_error(f"{field_prefix}.metric", "priority_hint must not be represented in metrics_used.")
                    else:
                        add_error(f"{field_prefix}.metric", "priority must not be represented in metrics_used.")

                if "value" in metric_item and not _is_metric_value_type(metric_item["value"]):
                    add_error(
                        f"{field_prefix}.value",
                        f"expected int/float/str/null, got {_type_name(metric_item['value'])}.",
                    )

                if "context" in metric_item:
                    context = metric_item["context"]
                    context_field = f"{field_prefix}.context"
                    if not isinstance(context, dict):
                        add_error(context_field, f"expected object, got {_type_name(context)}.")
                    else:
                        for key in REQUIRED_CONTEXT_KEYS:
                            if key not in context:
                                add_error(f"{context_field}.{key}", "required key is missing.")
                        for key in REQUIRED_CONTEXT_KEYS:
                            if key in context and not _is_str_or_none(context[key]):
                                add_error(
                                    f"{context_field}.{key}",
                                    f"expected str or null, got {_type_name(context[key])}.",
                                )
                            if key in context and isinstance(context[key], str) and _looks_like_priority_field(context[key]):
                                add_error(f"{context_field}.{key}", "priority metadata must not be stored in metrics_used context.")

    evidence_level = record.get("evidence_level")
    if "evidence_level" in record:
        if not isinstance(evidence_level, str):
            add_error(
                "evidence_level",
                f"expected one of ({_format_choices(ALLOWED_EVIDENCE_LEVELS)}), got {_type_name(evidence_level)}.",
            )
        elif evidence_level not in ALLOWED_EVIDENCE_LEVELS:
            add_error(
                "evidence_level",
                f"invalid value {evidence_level!r}; expected one of ({_format_choices(ALLOWED_EVIDENCE_LEVELS)}).",
            )

    if "limitation_or_counterpoint" in record:
        _validate_optional_string(
            value=record["limitation_or_counterpoint"],
            field="limitation_or_counterpoint",
            record_no=record_no,
            line_no=line_no,
            issues=issues,
        )

    next_data_needed = record.get("next_data_needed")
    if "next_data_needed" in record:
        _validate_optional_string(
            value=next_data_needed,
            field="next_data_needed",
            record_no=record_no,
            line_no=line_no,
            issues=issues,
        )
        if isinstance(next_data_needed, str) and _contains_priority_signal(next_data_needed):
            add_warning("next_data_needed", "priority-like content detected; keep priority in required_files.priority or priority_hint.")

    required_files = record.get("required_files")
    normalized_required_names: list[str] = []
    if "required_files" in record:
        if required_files is not None and not isinstance(required_files, list):
            add_error("required_files", f"expected list[object] or null, got {_type_name(required_files)}.")
        elif isinstance(required_files, list):
            for idx, item in enumerate(required_files):
                item_field = f"required_files[{idx}]"
                if not isinstance(item, dict):
                    add_error(item_field, f"expected object, got {_type_name(item)}.")
                    continue

                if "file_name" not in item:
                    add_error(f"{item_field}.file_name", "required key is missing.")
                else:
                    file_name = item.get("file_name")
                    if not isinstance(file_name, str):
                        add_error(f"{item_field}.file_name", f"expected str, got {_type_name(file_name)}.")
                    elif not file_name.strip():
                        add_error(f"{item_field}.file_name", "must not be empty.")
                    else:
                        normalized_required_names.append(file_name.strip())

                for optional_field in ("priority", "reason", "required_for"):
                    if optional_field in item and not _is_str_or_none(item[optional_field]):
                        add_error(
                            f"{item_field}.{optional_field}",
                            f"expected str or null, got {_type_name(item[optional_field])}.",
                        )

                if "priority" in item and isinstance(item["priority"], str) and normalize_priority(item["priority"]) is None:
                    add_error(f"{item_field}.priority", "must not be empty when provided.")

    raw_text = record.get("raw_text")
    if "raw_text" in record:
        if raw_text is None:
            pass
        elif isinstance(raw_text, str):
            if normalize_optional_text(raw_text) is None:
                add_warning("raw_text", "blank string is treated as missing.")
        elif isinstance(raw_text, list):
            for idx, item in enumerate(raw_text):
                if not isinstance(item, str):
                    add_error(f"raw_text[{idx}]", f"expected str, got {_type_name(item)}.")
            if normalize_raw_text(raw_text) is None:
                add_warning("raw_text", "all list entries are blank after normalization.")
            else:
                add_warning("raw_text", "list input will be normalized to a newline-joined string during ingest.")
        else:
            add_error("raw_text", f"expected str, list[str], or null, got {_type_name(raw_text)}.")

    evidence_refs = record.get("evidence_refs")
    normalized_evidence_refs: list[str] = []
    if "evidence_refs" in record:
        if evidence_refs is not None and not isinstance(evidence_refs, list):
            add_error("evidence_refs", f"expected list[str] or null, got {_type_name(evidence_refs)}.")
        elif isinstance(evidence_refs, list):
            for idx, item in enumerate(evidence_refs):
                if not isinstance(item, str):
                    add_error(f"evidence_refs[{idx}]", f"expected str, got {_type_name(item)}.")
                    continue
                if not item.strip():
                    add_warning(f"evidence_refs[{idx}]", "blank string is ignored.")
            normalized_refs = normalize_evidence_refs(evidence_refs)
            if normalized_refs is not None:
                normalized_evidence_refs = normalized_refs
            elif evidence_refs:
                add_warning("evidence_refs", "all entries are blank after normalization.")

    priority_hint = record.get("priority_hint")
    if "priority_hint" in record:
        _validate_optional_string(
            value=priority_hint,
            field="priority_hint",
            record_no=record_no,
            line_no=line_no,
            issues=issues,
        )
        if isinstance(priority_hint, str) and normalize_priority_hint(priority_hint) is None:
            add_warning("priority_hint", "blank string is treated as missing.")

    created_at = record.get("created_at")
    if "created_at" in record:
        _validate_optional_string(
            value=created_at,
            field="created_at",
            record_no=record_no,
            line_no=line_no,
            issues=issues,
        )
        if isinstance(created_at, str):
            normalized_created = normalize_created_at(created_at)
            if normalized_created is None:
                add_warning("created_at", "blank string is treated as missing.")
            elif not is_valid_created_at(normalized_created):
                add_warning("created_at", "unrecognized datetime format; ISO-8601 is recommended.")

    for field in (
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
    ):
        if field not in record:
            continue
        _validate_optional_string(
            value=record.get(field),
            field=field,
            record_no=record_no,
            line_no=line_no,
            issues=issues,
        )
        value = record.get(field)
        if isinstance(value, str) and normalize_optional_text(value) is None:
            add_warning(field, "blank string is treated as missing.")

    for field in ("exploration_axes", "fixed_conditions", "variable_conditions"):
        if field not in record:
            continue
        _validate_optional_string_list(
            value=record.get(field),
            field=field,
            record_no=record_no,
            line_no=line_no,
            issues=issues,
        )

    if normalized_required_names and not normalized_evidence_files:
        add_warning("required_files", "required_files is present but evidence_files is empty.")

    if normalized_evidence_refs and not normalized_evidence_files:
        add_warning("evidence_refs", "evidence_refs is present but evidence_files is empty.")

    if normalized_required_names and normalized_evidence_files:
        overlap = set(normalized_required_names) & set(normalized_evidence_files)
        if not overlap:
            add_warning("required_files", "no overlap with evidence_files; verify references are intentional.")

    return issues


def validate_records(records: Sequence[LoadedRecord]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for record in records:
        issues.extend(validate_record(record.payload, record_no=record.record_no, line_no=record.line_no))
    return issues


def _count_issues(issues: Sequence[ValidationIssue]) -> tuple[int, int]:
    errors = sum(1 for issue in issues if issue.level == "ERROR")
    warnings = sum(1 for issue in issues if issue.level == "WARNING")
    return errors, warnings


def run(path: Path, format_hint: str = "auto") -> int:
    records, load_issues = load_records(path=path, format_hint=format_hint)
    validation_issues = validate_records(records=records)
    issues = [*load_issues, *validation_issues]

    for issue in issues:
        print(issue.format())

    error_count, warning_count = _count_issues(issues)
    if error_count > 0:
        print(f"FAILED: {error_count} error(s), {warning_count} warning(s).")
        return 1

    print(f"OK: validated {len(records)} record(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arena.synthesis.validator")
    parser.add_argument("--path", required=True, help="Path to target .json or .jsonl file.")
    parser.add_argument(
        "--format",
        default="auto",
        choices=("auto", "json", "jsonl"),
        help="Input format hint. Default: auto (by extension).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(path=Path(args.path), format_hint=args.format)


if __name__ == "__main__":
    raise SystemExit(main())
