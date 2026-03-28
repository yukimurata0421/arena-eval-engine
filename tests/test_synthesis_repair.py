from __future__ import annotations

from arena.synthesis.repair import repair_json_text


def test_repair_json_text_fixes_doubled_quote_string_tokens() -> None:
    broken = "[\n  \"\"a\"\",\n  \"\"line \"quoted\" value\"\"\n]\n"
    result = repair_json_text(broken)
    assert result.applied_rules
    assert '  "a",' in result.text
    assert '\\"quoted\\"' in result.text


def test_repair_json_text_noop_when_not_matching() -> None:
    src = '["a", "b"]\n'
    result = repair_json_text(src)
    assert result.text.strip() == src.strip()


def test_repair_json_text_strips_code_fence_and_extracts_json_window() -> None:
    src = "prefix\n```json\n{\"claim\": \"x\"}\n```\nsuffix"
    result = repair_json_text(src)
    assert "strip_code_fence_wrapper:1" in result.applied_rules
    assert result.text.startswith("{")
    assert result.text.endswith("}")


def test_repair_json_text_converts_python_literals() -> None:
    src = '{"a": None, "b": True, "c": False}'
    result = repair_json_text(src)
    assert any(rule.startswith("python_literal_to_json_literal:") for rule in result.applied_rules)
    assert '"a": null' in result.text
    assert '"b": true' in result.text
    assert '"c": false' in result.text


def test_repair_json_text_repairs_double_wrapped_key_value_string() -> None:
    src = '{\n  "raw_text": ""line1"\\n"line2""\n}\n'
    result = repair_json_text(src)
    assert any(rule.startswith("double_wrapped_key_value_string:") for rule in result.applied_rules)
    assert '"raw_text": "line1\\nline2"' in result.text


def test_repair_json_text_repairs_inner_quote_string_token() -> None:
    src = '[\n  "bayesian_phase_results_cuda."Airspy Mini vs RTL-SDR""\n]\n'
    result = repair_json_text(src)
    assert any(rule.startswith("string_token_inner_quote_escape:") for rule in result.applied_rules)
    assert '\\"Airspy Mini vs RTL-SDR\\"' in result.text
