from __future__ import annotations

from datetime import datetime
from typing import Any

_PRIORITY_NORMALIZATION_MAP = {
    "a": "A",
    "b": "B",
    "c": "C",
    "priority a": "A",
    "priority b": "B",
    "priority c": "C",
    "high": "high",
    "medium": "medium",
    "low": "low",
}

_DATETIME_FORMAT_CANDIDATES = (
    "%Y/%m/%d",
    "%Y%m%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)


def normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def normalize_raw_text(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return normalize_optional_text(value)

    if not isinstance(value, list):
        return None

    normalized_lines: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = normalize_optional_text(item)
        if normalized is not None:
            normalized_lines.append(normalized)

    if not normalized_lines:
        return None
    return "\n".join(normalized_lines)


def normalize_priority(value: Any) -> str | None:
    normalized = normalize_optional_text(value)
    if normalized is None:
        return None

    key = " ".join(normalized.casefold().split())
    return _PRIORITY_NORMALIZATION_MAP.get(key, normalized)


def normalize_priority_hint(value: Any) -> str | None:
    normalized = normalize_optional_text(value)
    if normalized is None:
        return None

    priority_only = normalize_priority(normalized)
    if priority_only is not None and priority_only in {"A", "B", "C", "high", "medium", "low"}:
        return priority_only
    return normalized


def _parse_supported_datetime(value: str) -> datetime | None:
    candidate = value.strip()
    if not candidate:
        return None

    iso_candidate = candidate
    if iso_candidate.endswith("Z"):
        iso_candidate = f"{iso_candidate[:-1]}+00:00"

    try:
        return datetime.fromisoformat(iso_candidate)
    except ValueError:
        pass

    for fmt in _DATETIME_FORMAT_CANDIDATES:
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue

    return None


def normalize_created_at(value: Any) -> str | None:
    normalized = normalize_optional_text(value)
    if normalized is None:
        return None

    parsed = _parse_supported_datetime(normalized)
    if parsed is None:
        return normalized
    return parsed.isoformat()


def is_valid_created_at(value: str) -> bool:
    return _parse_supported_datetime(value) is not None


def normalize_evidence_refs(value: Any) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, str):
        normalized = normalize_optional_text(value)
        if normalized is None:
            return None
        return [normalized]

    if not isinstance(value, list):
        return None

    normalized_items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = normalize_optional_text(item)
        if normalized is not None:
            normalized_items.append(normalized)

    return normalized_items or None


def normalize_required_file_entry(value: Any) -> dict[str, str | None] | None:
    if not isinstance(value, dict):
        return None

    file_name = normalize_optional_text(value.get("file_name"))
    if file_name is None:
        return None

    return {
        "file_name": file_name,
        "priority": normalize_priority(value.get("priority")),
        "reason": normalize_optional_text(value.get("reason")),
        "required_for": normalize_optional_text(value.get("required_for")),
    }


def normalize_required_files(value: Any) -> list[dict[str, str | None]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None

    normalized_items: list[dict[str, str | None]] = []
    for item in value:
        normalized = normalize_required_file_entry(item)
        if normalized is None:
            continue
        normalized_items.append(normalized)

    return normalized_items or None


def normalize_optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None

    normalized_items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = normalize_optional_text(item)
        if normalized is not None:
            normalized_items.append(normalized)
    return normalized_items or None
