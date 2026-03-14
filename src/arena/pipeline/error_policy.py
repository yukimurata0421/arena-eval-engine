from __future__ import annotations

from typing import Sequence

from arena.pipeline.stages import RunRecord


def error_code_for_record(rec: RunRecord) -> str:
    status = rec.status.split("(")[0]
    if status in ("OK", "DRY", "SKIP"):
        return ""
    base = rec.step_code or f"S{rec.stage}-00"
    text = "\n".join(
        [
            rec.stderr_tail or "",
            rec.stdout_tail or "",
            "\n".join(rec.missing_outputs or []),
        ]
    ).lower()
    if status == "WARN":
        if "log_rotation_activity" in text:
            return f"{base}-W01"
        if "rate limit" in text:
            return f"{base}-W02"
        if "stale:" in text:
            return f"{base}-W03"
        return f"{base}-W99"
    if status == "FAIL_OUTPUT":
        if "stale:" in text:
            return f"{base}-E21"
        if "too small:" in text:
            return f"{base}-E22"
        if "empty dir" in text:
            return f"{base}-E23"
        return f"{base}-E20"
    if status == "TIMEOUT":
        return f"{base}-E31"
    if status == "NOT_FOUND":
        return f"{base}-E32"
    if status == "ERROR":
        return f"{base}-E33"
    if status == "FAIL":
        if "unicodeencodeerror" in text:
            return f"{base}-E11"
        if "rate limit" in text:
            return f"{base}-E12"
        return f"{base}-E10"
    return f"{base}-E99"


def summarize_issue_reason(rec: RunRecord) -> str:
    for text in (rec.stderr_tail or "", rec.stdout_tail or ""):
        if not text:
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue
        # Prefer lines that already explain operational degradation.
        for ln in reversed(lines):
            low = ln.lower()
            if any(k in low for k in ["fail", "warning", "warn", "timeout", "rate", "limit"]):
                return ln[:160]
        return lines[-1][:160]
    if rec.missing_outputs:
        return rec.missing_outputs[0][:160]
    return "details unavailable"


def recommended_actions(issues: Sequence[RunRecord]) -> list[str]:
    actions: list[str] = []
    codes = {(error_code_for_record(r) or "") for r in issues}

    # OpenSky rate-limit recovery guide.
    if "S1-06-W02" in codes or "S1-06-E12" in codes:
        actions.append(
            "OpenSky API rate limit reached. Wait 10-15 minutes and rerun only the OpenSky fetch:"
        )
        actions.append(
            "PowerShell: $env:OPENSKY_REFRESH_DAYS='1'; $env:OPENSKY_INCLUDE_TODAY='0'; $env:OPENSKY_MIN_DAILY_MOVEMENTS='0'; arena fetch-opensky"
        )
        actions.append(
            "After the fetch completes, refresh the comparison with: arena run --only 8"
        )

    if "S1-06-W99" in codes:
        actions.append(
            "OpenSky fetch ended with a warning. Check S1-06 in pipeline_runs.jsonl and rerun `arena fetch-opensky` if needed."
        )

    return actions
