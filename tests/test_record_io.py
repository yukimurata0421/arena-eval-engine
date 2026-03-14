from __future__ import annotations

from pathlib import Path

from arena.pipeline.record_io import (
    append_jsonl,
    build_config_snapshot_payload,
    build_run_record_payload,
)
from arena.pipeline.stages import RunRecord


def test_build_run_record_payload_contains_expected_keys() -> None:
    rec = RunRecord(
        ts_start="t0",
        ts_end="t1",
        backend="native",
        stage=1,
        label="label",
        script_rel="a.py",
        status="OK",
        elapsed_s=0.1,
        returncode=0,
        cmd=["python"],
        expected_outputs=[],
        outputs_ok=True,
        missing_outputs=[],
    )
    payload = build_run_record_payload(rec)
    assert payload["status"] == "OK"
    assert payload["stage"] == 1
    assert "error_code" in payload


def test_append_jsonl_writes_line(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    payload = build_config_snapshot_payload("ts", "native", {"a": 1})
    append_jsonl(p, payload)
    text = p.read_text(encoding="utf-8")
    assert '"status": "CONFIG"' in text
