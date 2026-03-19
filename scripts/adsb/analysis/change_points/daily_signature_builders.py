from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


@dataclass
class SeriesBuildResult:
    frame: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _to_local_date(record: dict[str, Any], timezone_name: str) -> datetime.date | None:
    tz = ZoneInfo(timezone_name) if ZoneInfo else None
    ts = record.get("ts")
    if ts is not None:
        try:
            ts_f = float(ts)
            if tz is not None:
                return datetime.fromtimestamp(ts_f, tz=timezone.utc).astimezone(tz).date()
            return pd.to_datetime(ts_f, unit="s", utc=True).tz_convert(timezone_name).date()
        except Exception:
            return None
    ts_iso = record.get("ts_iso")
    if ts_iso:
        try:
            dt = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if tz is not None:
                return dt.astimezone(tz).date()
            return pd.Timestamp(dt).tz_convert(timezone_name).date()
        except Exception:
            return None
    return None


def _iter_jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                yield path, payload


def _quantile_label(q: float) -> str:
    return f"q{int(round(q * 100))}"


def _extract_quantile_value(km: dict[str, Any], q: float) -> tuple[float | None, str]:
    key = f"p{int(round(q * 100))}"
    if key in km and km.get(key) is not None:
        try:
            return float(km[key]), "direct"
        except Exception:
            return None, "invalid"
    if abs(q - 0.99) < 1e-9:
        p95 = km.get("p95")
        vmax = km.get("max")
        if p95 is not None and vmax is not None:
            try:
                p95f = float(p95)
                vmaxf = float(vmax)
                # Linear tail interpolation between p95 and max (=q100 proxy).
                return p95f + ((0.99 - 0.95) / 0.05) * (vmaxf - p95f), "interpolated_from_p95_max"
            except Exception:
                return None, "invalid"
        if vmax is not None:
            try:
                return float(vmax), "fallback_max"
            except Exception:
                return None, "invalid"
    return None, "missing"


def load_daily_auc_series(auc_csv: Path) -> SeriesBuildResult:
    warnings: list[str] = []
    if not auc_csv.exists():
        warnings.append(f"auc_csv_not_found:{auc_csv}")
        return SeriesBuildResult(frame=pd.DataFrame(columns=["date", "daily_auc"]), warnings=warnings)
    try:
        df = pd.read_csv(auc_csv)
    except Exception as exc:
        warnings.append(f"auc_csv_read_failed:{exc}")
        return SeriesBuildResult(frame=pd.DataFrame(columns=["date", "daily_auc"]), warnings=warnings)

    if "date" not in df.columns or "auc_n_used" not in df.columns:
        warnings.append("auc_csv_missing_required_columns")
        return SeriesBuildResult(frame=pd.DataFrame(columns=["date", "daily_auc"]), warnings=warnings)

    out = df.loc[:, ["date", "auc_n_used"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["daily_auc"] = pd.to_numeric(out["auc_n_used"], errors="coerce")
    out = out.dropna(subset=["date", "daily_auc"]).sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].dt.normalize()
    out["auc_source"] = auc_csv.name
    out = out.loc[:, ["date", "daily_auc", "auc_source"]]

    stats = {
        "rows": int(len(out)),
        "date_min": out["date"].min().strftime("%Y-%m-%d") if not out.empty else "",
        "date_max": out["date"].max().strftime("%Y-%m-%d") if not out.empty else "",
    }
    return SeriesBuildResult(frame=out, warnings=warnings, stats=stats)


def build_daily_quantile_signature(
    dist_files: list[Path],
    quantiles: tuple[float, ...],
    timezone_name: str = "Asia/Tokyo",
) -> SeriesBuildResult:
    warnings: list[str] = []
    assumptions: list[str] = []
    if not dist_files:
        warnings.append("quantile_dist_files_empty")
        return SeriesBuildResult(frame=pd.DataFrame(columns=["date"]), warnings=warnings)

    labels = [_quantile_label(q) for q in quantiles]
    day_state: dict[pd.Timestamp, dict[str, float]] = {}
    method_counts: dict[str, int] = {}
    source_by_day: dict[pd.Timestamp, set[str]] = {}
    total_records = 0
    used_records = 0

    for source_path, record in _iter_jsonl(dist_files):
        total_records += 1
        src = str(record.get("src", ""))
        if src and src != "dist_1m":
            continue
        day = _to_local_date(record, timezone_name=timezone_name)
        if day is None:
            continue
        km = record.get("km")
        if not isinstance(km, dict):
            continue

        weight = km.get("n", record.get("n_used", 0))
        try:
            weight_f = float(weight)
        except Exception:
            continue
        if weight_f <= 0:
            continue

        used_records += 1
        day_key = pd.Timestamp(day)
        state = day_state.setdefault(day_key, {"weight_sum": 0.0, "minutes": 0.0})
        state["weight_sum"] += weight_f
        state["minutes"] += 1.0
        source_by_day.setdefault(day_key, set()).add(source_path.name)

        for q, label in zip(quantiles, labels):
            value, method = _extract_quantile_value(km=km, q=q)
            method_counts[method] = method_counts.get(method, 0) + 1
            if value is None:
                continue
            state[f"{label}_num"] = state.get(f"{label}_num", 0.0) + weight_f * float(value)
            state[f"{label}_den"] = state.get(f"{label}_den", 0.0) + weight_f

    rows: list[dict[str, Any]] = []
    for day_key in sorted(day_state.keys()):
        state = day_state[day_key]
        row: dict[str, Any] = {"date": day_key}
        for label in labels:
            den = state.get(f"{label}_den", 0.0)
            num = state.get(f"{label}_num", 0.0)
            row[label] = (num / den) if den > 0 else np.nan
        row["quantile_minutes_covered"] = int(state.get("minutes", 0.0))
        row["quantile_source"] = "|".join(sorted(source_by_day.get(day_key, set())))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        warnings.append("quantile_signature_empty_after_parse")
    else:
        out = out.sort_values("date").reset_index(drop=True)

    if method_counts.get("interpolated_from_p95_max", 0) > 0:
        assumptions.append(
            "Since p99 does not exist in dist_1m, q99 was approximated by linear interpolation of p95 and max."
        )
    if method_counts.get("fallback_max", 0) > 0:
        assumptions.append(
            "Q99 fell back to max for some records."
        )

    stats = {
        "records_total": total_records,
        "records_used": used_records,
        "days": int(len(out)),
        "method_counts": method_counts,
    }
    return SeriesBuildResult(frame=out, warnings=warnings, assumptions=assumptions, stats=stats)


_RANGE_RE = re.compile(r"^\s*([0-9.]+)\s*-\s*([0-9.]+)\s*km\s*$")
_PLUS_RE = re.compile(r"^\s*([0-9.]+)\+\s*km\s*$")


def _parse_bucket_label(label: str) -> tuple[float, float]:
    normalized = str(label).replace(" ", "")
    m1 = _RANGE_RE.match(normalized)
    if m1:
        return float(m1.group(1)), float(m1.group(2))
    m2 = _PLUS_RE.match(normalized)
    if m2:
        low = float(m2.group(1))
        return low, float("inf")
    raise ValueError(f"unsupported bucket label: {label}")


def _bucket_cdf_at_threshold(hist: dict[tuple[float, float], float], threshold_km: float) -> float:
    total = float(sum(hist.values()))
    if total <= 0:
        return np.nan
    covered = 0.0
    for (low, high), count in hist.items():
        if threshold_km <= low:
            continue
        if high <= threshold_km:
            covered += count
            continue
        if np.isfinite(high) and low < threshold_km < high:
            ratio = (threshold_km - low) / max(high - low, 1e-9)
            covered += max(0.0, min(1.0, ratio)) * count
    return covered / total


def build_daily_coverage_signature(
    signal_files: list[Path],
    coverage_grid_km: tuple[int, ...],
    timezone_name: str = "Asia/Tokyo",
) -> SeriesBuildResult:
    warnings: list[str] = []
    assumptions: list[str] = []
    if not signal_files:
        warnings.append("coverage_signal_files_empty")
        return SeriesBuildResult(frame=pd.DataFrame(columns=["date"]), warnings=warnings)

    per_day_hist: dict[pd.Timestamp, dict[tuple[float, float], float]] = {}
    per_day_sources: dict[pd.Timestamp, set[str]] = {}
    records_total = 0
    records_used = 0

    for source_path, record in _iter_jsonl(signal_files):
        records_total += 1
        src = str(record.get("src", ""))
        if src and src != "dist_signal_stats":
            continue
        day = _to_local_date(record, timezone_name=timezone_name)
        if day is None:
            continue
        buckets = record.get("buckets")
        if not isinstance(buckets, dict):
            continue
        day_key = pd.Timestamp(day)
        hist = per_day_hist.setdefault(day_key, {})
        per_day_sources.setdefault(day_key, set()).add(source_path.name)
        any_bucket = False
        for label, payload in buckets.items():
            if not isinstance(payload, dict):
                continue
            n_samples = payload.get("n_samples")
            try:
                n_samples_f = float(n_samples)
            except Exception:
                continue
            if n_samples_f <= 0:
                continue
            try:
                low, high = _parse_bucket_label(label)
            except Exception:
                continue
            hist[(low, high)] = hist.get((low, high), 0.0) + n_samples_f
            any_bucket = True
        if any_bucket:
            records_used += 1

    rows: list[dict[str, Any]] = []
    for day_key in sorted(per_day_hist.keys()):
        hist = per_day_hist[day_key]
        row: dict[str, Any] = {"date": day_key}
        for km in coverage_grid_km:
            row[f"coverage_{int(km)}km"] = _bucket_cdf_at_threshold(hist, float(km))
        row["coverage_source"] = "|".join(sorted(per_day_sources.get(day_key, set())))
        row["coverage_total_samples"] = int(sum(hist.values()))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        warnings.append("coverage_signature_empty_after_parse")
    else:
        out = out.sort_values("date").reset_index(drop=True)

    assumptions.append(
        "Coverage sums up the distance buckets of dist_signal_stats on a daily basis, and if the threshold falls within the bucket, linear interpolation is performed assuming uniform distribution."
    )
    stats = {"records_total": records_total, "records_used": records_used, "days": int(len(out))}
    return SeriesBuildResult(frame=out, warnings=warnings, assumptions=assumptions, stats=stats)
