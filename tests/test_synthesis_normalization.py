from __future__ import annotations

from arena.synthesis.normalization import (
    is_valid_created_at,
    normalize_created_at,
    normalize_evidence_refs,
    normalize_optional_string_list,
    normalize_priority,
    normalize_raw_text,
)


def test_normalize_priority_supports_common_labels() -> None:
    assert normalize_priority("A") == "A"
    assert normalize_priority("priority b") == "B"
    assert normalize_priority("High") == "high"
    assert normalize_priority("  ") is None


def test_normalize_created_at_supports_safe_formats() -> None:
    assert normalize_created_at("2026-03-22T01:02:03Z") == "2026-03-22T01:02:03+00:00"
    assert normalize_created_at("20260322") == "2026-03-22T00:00:00"
    assert normalize_created_at("2026/03/22 12:34:56") == "2026-03-22T12:34:56"


def test_normalize_created_at_keeps_unparseable_value() -> None:
    value = normalize_created_at("22-03-2026")
    assert value == "22-03-2026"
    assert is_valid_created_at(value) is False


def test_normalize_evidence_refs_filters_blank_values() -> None:
    assert normalize_evidence_refs([" ref1 ", "", "  ", "ref2"]) == ["ref1", "ref2"]
    assert normalize_evidence_refs(None) is None


def test_normalize_raw_text_supports_string_and_list() -> None:
    assert normalize_raw_text("  one line  ") == "one line"
    assert normalize_raw_text([" line1 ", "", "line2"]) == "line1\nline2"
    assert normalize_raw_text(["", "  "]) is None


def test_normalize_optional_string_list_filters_blank_values() -> None:
    assert normalize_optional_string_list([" axis1 ", "", "axis2"]) == ["axis1", "axis2"]
    assert normalize_optional_string_list(None) is None
