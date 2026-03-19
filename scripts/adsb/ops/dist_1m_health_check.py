#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from arena.lib.config import get_quality_thresholds
from arena.lib.paths import DATA_DIR, OUTPUT_DIR

from arena.log import get_script_logger



log = get_script_logger(__name__)
@dataclass
class CheckResult:
    name: str
    status: str  # ok | warn | fail
    message: str
    metrics: dict[str, Any]


def _parse_ts(rec: dict[str, Any]) -> float | None:
    ts = rec.get("ts")
    if ts is None and rec.get("ts_iso"):
        try:
            return datetime.fromisoformat(str(rec["ts_iso"]).replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return float(str(ts))
    except Exception:
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
        except Exception:
            return None


def _status_rank(s: str) -> int:
    if s == "fail":
        return 2
    if s == "warn":
        return 1
    return 0


def _pick_status(a: str, b: str) -> str:
    return a if _status_rank(a) >= _status_rank(b) else b


def evaluate(
    dist_file: Path,
    archive_dir: Path,
    max_staleness_minutes: int,
    min_minutes_yesterday: int,
    recent_gap_max_minutes: int,
    recent_window_hours: int,
    rotation_warn_hours: int,
    rotation_fail_hours: int,
    now_ts: float | None = None,
) -> tuple[str, list[CheckResult], dict[str, Any], list[dict[str, Any]]]:
    now_ts = float(now_ts) if now_ts is not None else time.time()
    checks: list[CheckResult] = []
    daily_minute_sets: dict[str, set[int]] = defaultdict(set)
    recent_minutes: list[int] = []

    total_lines = 0
    bad_lines = 0
    latest_ts: float | None = None
    earliest_ts: float | None = None

    if not dist_file.exists() or dist_file.stat().st_size <= 0:
        checks.append(
            CheckResult(
                name="dist_file_exists",
                status="fail",
                message=f"dist file missing or empty: {dist_file}",
                metrics={},
            )
        )
        summary = {
            "dist_file": str(dist_file),
            "archive_dir": str(archive_dir),
            "total_lines": 0,
            "bad_lines": 0,
        }
        return "fail", checks, summary, []

    recent_cutoff_minute = int((now_ts - recent_window_hours * 3600) // 60)
    with dist_file.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                rec = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            src = str(rec.get("src", "") or "")
            if src and src != "dist_1m":
                continue

            ts = _parse_ts(rec)
            if ts is None:
                bad_lines += 1
                continue

            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
            if earliest_ts is None or ts < earliest_ts:
                earliest_ts = ts

            minute_id = int(ts // 60)
            local_day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            daily_minute_sets[local_day].add(minute_id)

            if minute_id >= recent_cutoff_minute:
                recent_minutes.append(minute_id)

    if latest_ts is None:
        checks.append(
            CheckResult(
                name="dist_records_valid",
                status="fail",
                message="no valid dist_1m records with timestamp",
                metrics={"total_lines": total_lines, "bad_lines": bad_lines},
            )
        )
        summary = {
            "dist_file": str(dist_file),
            "archive_dir": str(archive_dir),
            "total_lines": total_lines,
            "bad_lines": bad_lines,
        }
        return "fail", checks, summary, []

    # 1) staleness check
    age_min = (now_ts - latest_ts) / 60.0
    stale_status = "ok" if age_min <= max_staleness_minutes else "fail"
    checks.append(
        CheckResult(
            name="latest_record_freshness",
            status=stale_status,
            message=(
                f"latest record age={age_min:.1f}m "
                f"(threshold={max_staleness_minutes}m)"
            ),
            metrics={
                "latest_ts_iso": datetime.fromtimestamp(latest_ts).isoformat(),
                "age_minutes": round(age_min, 2),
                "max_staleness_minutes": max_staleness_minutes,
            },
        )
    )

    # 2) yesterday coverage check
    today_local = datetime.fromtimestamp(now_ts).date()
    yday = today_local - timedelta(days=1)
    ykey = yday.isoformat()
    y_minutes = len(daily_minute_sets.get(ykey, set()))
    y_status = "ok" if y_minutes >= min_minutes_yesterday else "fail"
    checks.append(
        CheckResult(
            name="yesterday_minutes_covered",
            status=y_status,
            message=(
                f"{ykey}: minutes_covered={y_minutes} "
                f"(threshold={min_minutes_yesterday})"
            ),
            metrics={
                "date": ykey,
                "minutes_covered": y_minutes,
                "min_minutes_yesterday": min_minutes_yesterday,
            },
        )
    )

    # 3) recent gap check
    gap_status = "ok"
    gap_msg = "recent minute stream has no critical gap"
    max_gap = 0
    if recent_minutes:
        uniq = sorted(set(recent_minutes))
        for i in range(1, len(uniq)):
            gap = uniq[i] - uniq[i - 1]
            if gap > max_gap:
                max_gap = gap
        tail_gap = int(now_ts // 60) - uniq[-1]
        if tail_gap > max_gap:
            max_gap = tail_gap
        if max_gap > recent_gap_max_minutes:
            gap_status = "fail"
            gap_msg = (
                f"max recent minute gap={max_gap}m "
                f"(threshold={recent_gap_max_minutes}m)"
            )
    else:
        gap_status = "fail"
        gap_msg = "no records in recent window"
    checks.append(
        CheckResult(
            name="recent_gap_check",
            status=gap_status,
            message=gap_msg,
            metrics={
                "recent_window_hours": recent_window_hours,
                "max_gap_minutes": int(max_gap),
                "threshold_minutes": recent_gap_max_minutes,
            },
        )
    )

    # 4) archive rotation check (warn/fail by age)
    rot_status = "ok"
    rot_msg = "archive rotation looks active"
    rot_metrics: dict[str, Any] = {
        "archive_dir_exists": archive_dir.exists(),
        "latest_archive_mtime_iso": None,
        "latest_archive_age_hours": None,
        "warn_hours": rotation_warn_hours,
        "fail_hours": rotation_fail_hours,
    }
    if rotation_warn_hours <= 0 and rotation_fail_hours <= 0:
        rot_status = "ok"
        rot_msg = "archive rotation check disabled"
    elif archive_dir.exists():
        files = [p for p in archive_dir.glob("*.jsonl*") if p.is_file()]
        if files:
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            age_h = (now_ts - latest_file.stat().st_mtime) / 3600.0
            rot_metrics["latest_archive_file"] = str(latest_file)
            rot_metrics["latest_archive_mtime_iso"] = datetime.fromtimestamp(
                latest_file.stat().st_mtime
            ).isoformat()
            rot_metrics["latest_archive_age_hours"] = round(age_h, 2)
            if rotation_fail_hours > 0 and age_h > rotation_fail_hours:
                rot_status = "fail"
                rot_msg = (
                    f"archive not updated for {age_h:.1f}h "
                    f"(fail>{rotation_fail_hours}h)"
                )
            elif rotation_warn_hours > 0 and age_h > rotation_warn_hours:
                rot_status = "warn"
                rot_msg = (
                    f"archive not updated for {age_h:.1f}h "
                    f"(warn>{rotation_warn_hours}h)"
                )
        else:
            rot_status = "warn"
            rot_msg = "archive dir has no jsonl files"
    else:
        rot_status = "warn"
        rot_msg = "archive dir does not exist"
    checks.append(
        CheckResult(
            name="log_rotation_activity",
            status=rot_status,
            message=rot_msg,
            metrics=rot_metrics,
        )
    )

    # daily compact table (last 14 local days)
    coverage_rows: list[dict[str, Any]] = []
    all_days = sorted(daily_minute_sets.keys())
    for d in all_days[-14:]:
        minutes = len(daily_minute_sets[d])
        coverage_rows.append(
            {
                "date": d,
                "minutes_covered": minutes,
                "coverage_ratio": round(minutes / 1440.0, 4),
                "is_yesterday": 1 if d == ykey else 0,
                "is_today": 1 if d == today_local.isoformat() else 0,
            }
        )

    overall = "ok"
    for c in checks:
        overall = _pick_status(overall, c.status)

    summary = {
        "dist_file": str(dist_file),
        "archive_dir": str(archive_dir),
        "total_lines": total_lines,
        "bad_lines": bad_lines,
        "latest_ts_iso": datetime.fromtimestamp(latest_ts).isoformat(),
        "earliest_ts_iso": datetime.fromtimestamp(earliest_ts).isoformat(),
        "days_seen": len(all_days),
    }
    return overall, checks, summary, coverage_rows


def _write_daily_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "minutes_covered", "coverage_ratio", "is_yesterday", "is_today"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_arg_parser() -> argparse.ArgumentParser:
    _, min_minutes = get_quality_thresholds()
    default_dist = DATA_DIR / "dist_1m.jsonl"
    default_archive = DATA_DIR / "raw" / "past_log"
    default_json = OUTPUT_DIR / "performance" / "dist_1m_health_latest.json"
    default_csv = OUTPUT_DIR / "performance" / "dist_1m_health_daily.csv"

    ap = argparse.ArgumentParser(
        description="Operational health check for dist_1m collector and log rotation."
    )
    ap.add_argument("--dist-file", default=str(default_dist))
    ap.add_argument("--archive-dir", default=str(default_archive))
    ap.add_argument("--output-json", default=str(default_json))
    ap.add_argument("--output-daily-csv", default=str(default_csv))
    ap.add_argument("--max-staleness-minutes", type=int, default=20)
    ap.add_argument("--min-minutes-yesterday", type=int, default=int(min_minutes))
    ap.add_argument("--recent-gap-max-minutes", type=int, default=15)
    ap.add_argument("--recent-window-hours", type=int, default=24)
    ap.add_argument("--rotation-warn-hours", type=int, default=72)
    ap.add_argument("--rotation-fail-hours", type=int, default=168)
    ap.add_argument("--now-ts", type=float, default=None, help="optional epoch seconds for reproducible tests")
    return ap


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()

    dist_file = Path(args.dist_file)
    archive_dir = Path(args.archive_dir)
    out_json = Path(args.output_json)
    out_daily = Path(args.output_daily_csv)

    overall, checks, summary, rows = evaluate(
        dist_file=dist_file,
        archive_dir=archive_dir,
        max_staleness_minutes=int(args.max_staleness_minutes),
        min_minutes_yesterday=int(args.min_minutes_yesterday),
        recent_gap_max_minutes=int(args.recent_gap_max_minutes),
        recent_window_hours=int(args.recent_window_hours),
        rotation_warn_hours=int(args.rotation_warn_hours),
        rotation_fail_hours=int(args.rotation_fail_hours),
        now_ts=args.now_ts,
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": overall,
        "summary": summary,
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "message": c.message,
                "metrics": c.metrics,
            }
            for c in checks
        ],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_daily_csv(out_daily, rows)

    log.info(f"[dist_1m_health] status={overall}")
    for c in checks:
        log.info(f"  - {c.status.upper():4} {c.name}: {c.message}")
    log.info(f"  wrote: {out_json}")
    log.info(f"  wrote: {out_daily}")

    return 0 if overall in ("ok", "warn") else 1


if __name__ == "__main__":
    raise SystemExit(main())
