from __future__ import annotations

from arena.pipeline.error_policy import (
    error_code_for_record,
    recommended_actions,
    summarize_issue_reason,
)
from arena.pipeline.stages import RunRecord


def _rec(**overrides) -> RunRecord:
    data = dict(
        ts_start="t0",
        ts_end="t1",
        backend="native",
        stage=1,
        label="label",
        script_rel="script.py",
        status="OK",
        elapsed_s=0.1,
        returncode=0,
        cmd=["python", "script.py"],
        expected_outputs=[],
        outputs_ok=True,
        missing_outputs=[],
        step_code="S1-06",
        error_code="",
        stderr_tail="",
        stdout_tail="",
    )
    data.update(overrides)
    return RunRecord(**data)


def test_error_code_warn_rate_limit() -> None:
    rec = _rec(status="WARN", stderr_tail="rate limit reached")
    assert error_code_for_record(rec) == "S1-06-W02"


def test_error_code_fail_output_too_small() -> None:
    rec = _rec(status="FAIL_OUTPUT", missing_outputs=["x.csv (too small: 12 bytes)"])
    assert error_code_for_record(rec) == "S1-06-E22"


def test_summarize_issue_reason_prefers_warning_line() -> None:
    rec = _rec(
        status="WARN",
        stderr_tail="line1\nnormal line\nWARNING: something degraded",
    )
    assert summarize_issue_reason(rec).startswith("WARNING:")


def test_recommended_actions_for_opensky_rate_limit() -> None:
    rec = _rec(status="WARN", stderr_tail="rate limit", step_code="S1-06")
    actions = recommended_actions([rec])
    assert any("OpenSky API" in a for a in actions)
    assert any("arena fetch-opensky" in a for a in actions)


def test_warn_without_detail_uses_w99_and_generic_action() -> None:
    rec = _rec(status="WARN", stderr_tail="", stdout_tail="", step_code="S1-06")
    assert error_code_for_record(rec) == "S1-06-W99"
    actions = recommended_actions([rec])
    assert any("OpenSky fetch ended with a warning" in a for a in actions)
