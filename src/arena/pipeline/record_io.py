from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.pipeline.stages import RunRecord


def build_run_record_payload(rec: RunRecord) -> dict[str, Any]:
    return {
        "ts_start": rec.ts_start,
        "ts_end": rec.ts_end,
        "backend": rec.backend,
        "stage": rec.stage,
        "label": rec.label,
        "script": rec.script_rel,
        "status": rec.status,
        "step_code": rec.step_code,
        "error_code": rec.error_code,
        "elapsed_s": rec.elapsed_s,
        "returncode": rec.returncode,
        "cmd": rec.cmd,
        "expected_outputs": rec.expected_outputs,
        "outputs_ok": rec.outputs_ok,
        "missing_outputs": rec.missing_outputs,
        "stderr_tail": rec.stderr_tail,
        "stdout_tail": rec.stdout_tail,
    }


def build_config_snapshot_payload(ts: str, backend_desc: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": ts,
        "status": "CONFIG",
        "backend": backend_desc,
        "config_snapshot": snapshot,
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
