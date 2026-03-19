from __future__ import annotations

from collections.abc import Sequence

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
        if "rate limit" in text or "レート制限" in text:
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
        if "rate limit" in text or "レート制限" in text:
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
            if any(k in low for k in ["fail", "warning", "warn", "timeout", "rate", "limit", "中断", "到達"]):
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
        actions.append("OpenSky API のレート制限に到達。10〜15分待ってから OpenSky 取得のみ再実行:")
        actions.append(
            "PowerShell: $env:OPENSKY_REFRESH_DAYS='1'; $env:OPENSKY_INCLUDE_TODAY='0'; $env:OPENSKY_MIN_DAILY_MOVEMENTS='0'; arena fetch-opensky"
        )
        actions.append("取得完了後、比較を更新するなら: arena run --only 8")

    if "S1-06-W99" in codes:
        actions.append("OpenSky 取得が警告終了（詳細は pipeline_runs.jsonl の S1-06 を確認）。必要なら arena fetch-opensky を単体実行。")

    return actions
