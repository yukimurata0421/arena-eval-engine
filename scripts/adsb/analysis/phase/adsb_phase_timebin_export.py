from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from arena.lib.paths import ADSB_DAILY_SUMMARY_V2, OUTPUT_DIR
from arena.lib.phase_config import Event, get_config

from arena.log import get_script_logger



log = get_script_logger(__name__)
PERFORMANCE_DIR = Path(OUTPUT_DIR) / "performance"
TIMEBIN_SUMMARY_PATH = Path(OUTPUT_DIR) / "time_resolved" / "adsb_timebin_summary.csv"
MAPPING_CSV_PATH = PERFORMANCE_DIR / "phase_config_daily_mapping.csv"
MAPPING_JSON_PATH = PERFORMANCE_DIR / "phase_config_daily_mapping.json"
PHASE_TIMEBIN_CSV_PATH = PERFORMANCE_DIR / "phase_timebin_summary.csv"
PHASE_TIMEBIN_JSON_PATH = PERFORMANCE_DIR / "phase_timebin_summary.json"
REPORT_PATH = PERFORMANCE_DIR / "phase_timebin_export_report.txt"


@dataclass
class PhaseState:
    phase_id: int
    phase_name: str
    phase_label: str
    phase_start_date: date
    phase_end_date: date | None
    start_event_label: str


def _to_date(value: Any) -> date | None:
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def _fmt_date(value: date | None) -> str:
    return value.isoformat() if isinstance(value, date) else ""


def _read_csv_with_date(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df[df["date"].notna()].copy()
    return df


def _build_phase_states(events: list[Event], default_hardware: str, phase_names: dict[int, str]) -> list[PhaseState]:
    hardware_events = [e for e in events if e.hardware]
    if not hardware_events:
        fallback_start = date.today()
        return [
            PhaseState(
                phase_id=0,
                phase_name=default_hardware or "unknown",
                phase_label=phase_names.get(0, default_hardware or "unknown"),
                phase_start_date=fallback_start,
                phase_end_date=None,
                start_event_label="fallback_no_hardware_event",
            )
        ]

    states: list[PhaseState] = []
    for idx, ev in enumerate(hardware_events):
        start = _to_date(ev.date)
        if start is None:
            continue
        if idx + 1 < len(hardware_events):
            next_start = _to_date(hardware_events[idx + 1].date)
            end = (next_start - timedelta(days=1)) if next_start else None
        else:
            end = None
        states.append(
            PhaseState(
                phase_id=idx,
                phase_name=ev.hardware,
                phase_label=phase_names.get(idx, ev.label or ev.hardware or f"phase_{idx}"),
                phase_start_date=start,
                phase_end_date=end,
                start_event_label=ev.label,
            )
        )

    if not states:
        fallback_start = date.today()
        states.append(
            PhaseState(
                phase_id=0,
                phase_name=default_hardware or "unknown",
                phase_label=phase_names.get(0, default_hardware or "unknown"),
                phase_start_date=fallback_start,
                phase_end_date=None,
                start_event_label="fallback_parse_failed",
            )
        )

    return states


def _phase_for_day(target: date, states: list[PhaseState]) -> PhaseState:
    current = states[0]
    for st in states:
        if st.phase_start_date <= target and (st.phase_end_date is None or target <= st.phase_end_date):
            current = st
            break
        if st.phase_start_date <= target:
            current = st
    return current


def _latest_event_for_day(target: date, events: list[Event]) -> Event | None:
    latest: Event | None = None
    for ev in events:
        d = _to_date(ev.date)
        if d is None:
            continue
        if d <= target:
            latest = ev
        else:
            break
    return latest


def _extract_gain_profile(label: str) -> tuple[str, bool]:
    if not label:
        return "", False
    matches = re.findall(r"(Gain[^,|/;]*)", label, flags=re.IGNORECASE)
    if not matches:
        return "", False
    cleaned = [m.strip() for m in matches if m.strip()]
    if not cleaned:
        return "", False
    return " / ".join(dict.fromkeys(cleaned)), False


def _extract_filter_type(label: str) -> tuple[str, bool]:
    if not label:
        return "", False
    tokens = re.findall(r"(-f\s*\d+)", label, flags=re.IGNORECASE)
    if tokens:
        return ", ".join(dict.fromkeys(t.replace(" ", "") for t in tokens)), False
    if "filter" in label.lower():
        return "filter_changed", True
    return "", False


def _extract_cable_type(label: str, hardware: str) -> tuple[str, bool]:
    low = (label or "").lower()
    if "5d-fb" in low:
        return "5D-FB", False
    if "2.5ds-qfb" in low:
        return "2.5DS-QFB", False
    if "cable" in low:
        return "cable_changed", True
    if "plus_cable" in (hardware or "").lower():
        return "cable_attached", True
    return "", False


def _extract_adapter_type(label: str, hardware: str) -> tuple[str, bool]:
    low = (label or "").lower()
    m = re.search(r"adapter[^()]*\(([^)]+)\)", label or "", flags=re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip(), False
    if "adapter" in low or "adapter" in (hardware or "").lower():
        return "adapter_changed", True
    return "", False


def _extract_sdr_type(hardware: str) -> tuple[str, bool]:
    hw = (hardware or "").lower()
    if "rtl" in hw:
        return "rtl-sdr", False
    if "airspy" in hw:
        return "airspy", False
    if hw:
        return hw, True
    return "unknown", True


def _build_hardware_stack(fields: dict[str, str]) -> str:
    parts = []
    for key in ["sdr_type", "gain_profile", "cable_type", "adapter_type", "filter_type"]:
        val = (fields.get(key) or "").strip()
        if val:
            parts.append(f"{key}={val}")
    return " | ".join(parts)


def _config_hash(fields: dict[str, str]) -> str:
    raw = "|".join((fields.get(k, "") or "").strip() for k in ["phase_name", "sdr_type", "gain_profile", "cable_type", "adapter_type", "filter_type"])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def build_phase_config_daily_mapping(daily_df: pd.DataFrame, timebin_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cfg = get_config()
    events = sorted(cfg.events, key=lambda e: e.date)
    states = _build_phase_states(events=events, default_hardware=cfg.default_hardware, phase_names=cfg.phase_names)
    warnings: list[str] = []

    daily_dates = set(daily_df["date"].tolist()) if ("date" in daily_df.columns and not daily_df.empty) else set()
    timebin_dates = set(timebin_df["date"].tolist()) if ("date" in timebin_df.columns and not timebin_df.empty) else set()
    phase_dates = {st.phase_start_date for st in states}
    date_candidates = sorted(d for d in (daily_dates | timebin_dates | phase_dates) if isinstance(d, date))

    if date_candidates:
        start = min(date_candidates)
        end = max(date_candidates)
    else:
        start = date.today()
        end = date.today()
        warnings.append("no_input_dates_found; fallback=today")

    all_dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    transition_dates = {st.phase_start_date for st in states}

    rows: list[dict[str, Any]] = []
    for d in all_dates:
        st = _phase_for_day(d, states)
        ev = _latest_event_for_day(d, events)
        event_label = ev.label if ev else st.start_event_label

        sdr_type, sdr_inferred = _extract_sdr_type(st.phase_name)
        gain_profile, gain_inferred = _extract_gain_profile(event_label)
        cable_type, cable_inferred = _extract_cable_type(event_label, st.phase_name)
        adapter_type, adapter_inferred = _extract_adapter_type(event_label, st.phase_name)
        filter_type, filter_inferred = _extract_filter_type(event_label)

        inferred_fields = []
        if sdr_inferred:
            inferred_fields.append("sdr_type")
        if gain_inferred:
            inferred_fields.append("gain_profile")
        if cable_inferred:
            inferred_fields.append("cable_type")
        if adapter_inferred:
            inferred_fields.append("adapter_type")
        if filter_inferred:
            inferred_fields.append("filter_type")

        core_fields = {
            "phase_name": st.phase_name,
            "sdr_type": sdr_type,
            "gain_profile": gain_profile,
            "cable_type": cable_type,
            "adapter_type": adapter_type,
            "filter_type": filter_type,
        }
        stack = _build_hardware_stack(core_fields)
        row = {
            "date": _fmt_date(d),
            "phase_id": st.phase_id,
            "phase_name": st.phase_name,
            "phase_label": st.phase_label,
            "sdr_type": sdr_type,
            "gain": gain_profile,
            "gain_profile": gain_profile,
            "cable_type": cable_type,
            "adapter_type": adapter_type,
            "filter_type": filter_type,
            "notes": event_label,
            "source": "phase_config.events",
            "phase_start_date": _fmt_date(st.phase_start_date),
            "phase_end_date": _fmt_date(st.phase_end_date),
            "hardware_stack": stack,
            "config_hash": _config_hash(core_fields),
            "is_transition_day": int(d in transition_dates),
            "has_daily_summary_v2": int(d in daily_dates),
            "has_timebin_summary": int(d in timebin_dates),
            "inferred_fields": ",".join(inferred_fields),
        }
        rows.append(row)

    mapping = pd.DataFrame(rows)
    watched = ["phase_id", "phase_name", "sdr_type", "gain_profile", "cable_type", "adapter_type", "filter_type", "config_hash"]
    # ベクトル化: per-cell ループ → shift() による一括比較
    _str = mapping[watched].astype(str)
    _prev = _str.shift(1)
    _changed_mask = _str != _prev
    _changed_mask.iloc[0] = False  # 最初の行は比較対象なし
    mapping["changed_from_previous_day"] = _changed_mask.any(axis=1).astype(int)
    mapping["changed_fields"] = _changed_mask.apply(
        lambda row: ",".join(col for col, v in row.items() if v), axis=1
    )

    return mapping, warnings


def build_phase_timebin_summary(timebin_df: pd.DataFrame, mapping_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    out_cols = [
        "phase_id",
        "phase_name",
        "phase_label",
        "time_bin",
        "n_days",
        "n_rows",
        "n_used",
        "usable_days",
        "dropped_days",
        "mean_auc_n_used",
        "median_auc_n_used",
        "std_auc_n_used",
        "q25_auc_n_used",
        "q75_auc_n_used",
        "min_auc_n_used",
        "max_auc_n_used",
        "mean_minutes_covered",
        "median_minutes_covered",
        "mean_capture_ratio",
        "phase_vs_baseline_diff_pct",
        "phase_vs_baseline_ratio",
        "phase_start_date",
        "phase_end_date",
    ]
    if timebin_df.empty or "date" not in timebin_df.columns:
        warnings.append("timebin_summary_missing_or_empty")
        return pd.DataFrame(columns=out_cols), warnings

    wb = timebin_df.copy()
    wb["date"] = pd.to_datetime(wb["date"], errors="coerce").dt.date
    wb = wb[wb["date"].notna()].copy()
    if wb.empty:
        warnings.append("timebin_summary_has_no_valid_date")
        return pd.DataFrame(columns=out_cols), warnings

    map_use = mapping_df[["date", "phase_id", "phase_name", "phase_label"]].copy()
    map_use["date"] = pd.to_datetime(map_use["date"], errors="coerce").dt.date
    merged = wb.merge(map_use, on="date", how="left")

    for col in ["auc_sum", "minutes"]:
        if col not in merged.columns:
            merged[col] = pd.NA
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    group_cols = ["phase_id", "phase_name", "phase_label", "time_bin"]
    grp = merged.groupby(group_cols, dropna=False)
    # lambda を使わず Cython 高速パスを維持するため quantile は別途計算して merge
    agg = (
        grp.agg(
            n_days=("date", "nunique"),
            n_rows=("date", "size"),
            n_used=("minutes", "sum"),
            mean_auc_n_used=("auc_sum", "mean"),
            median_auc_n_used=("auc_sum", "median"),
            std_auc_n_used=("auc_sum", "std"),
            min_auc_n_used=("auc_sum", "min"),
            max_auc_n_used=("auc_sum", "max"),
            mean_minutes_covered=("minutes", "mean"),
            median_minutes_covered=("minutes", "median"),
        )
    )
    agg["q25_auc_n_used"] = grp["auc_sum"].quantile(0.25)
    agg["q75_auc_n_used"] = grp["auc_sum"].quantile(0.75)
    agg = agg.reset_index()

    agg["usable_days"] = agg["n_days"]
    phase_total_days = mapping_df.groupby("phase_id", dropna=False)["date"].nunique().to_dict()
    agg["dropped_days"] = agg.apply(
        lambda r: max(int(phase_total_days.get(r["phase_id"], r["n_days"])) - int(r["n_days"]), 0),
        axis=1,
    )

    if "capture_ratio" in merged.columns:
        cap = (
            merged.groupby(group_cols, dropna=False)["capture_ratio"]
            .mean()
            .reset_index(name="mean_capture_ratio")
        )
        agg = agg.merge(cap, on=group_cols, how="left")
    else:
        agg["mean_capture_ratio"] = pd.NA

    valid_phase_ids = sorted([x for x in agg["phase_id"].dropna().unique()])
    baseline_id = int(valid_phase_ids[0]) if valid_phase_ids else 0
    baseline = agg[agg["phase_id"] == baseline_id][["time_bin", "mean_auc_n_used"]].rename(
        columns={"mean_auc_n_used": "baseline_auc"}
    )
    agg = agg.merge(baseline, on="time_bin", how="left")
    agg["phase_vs_baseline_ratio"] = agg["mean_auc_n_used"] / agg["baseline_auc"]
    agg["phase_vs_baseline_diff_pct"] = (agg["phase_vs_baseline_ratio"] - 1.0) * 100.0
    agg.loc[agg["baseline_auc"].isna() | (agg["baseline_auc"] == 0), ["phase_vs_baseline_ratio", "phase_vs_baseline_diff_pct"]] = pd.NA
    agg = agg.drop(columns=["baseline_auc"])

    phase_ranges = (
        mapping_df.groupby(["phase_id", "phase_name", "phase_label"], dropna=False)
        .agg(
            phase_start_date=("date", "min"),
            phase_end_date=("date", "max"),
        )
        .reset_index()
    )
    agg = agg.merge(phase_ranges, on=["phase_id", "phase_name", "phase_label"], how="left")
    agg = agg[out_cols].sort_values(["phase_id", "time_bin"]).reset_index(drop=True)
    return agg, warnings


def _write_json_records(path: Path, df: pd.DataFrame) -> None:
    records = df.to_dict(orient="records")
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    daily_df = _read_csv_with_date(Path(ADSB_DAILY_SUMMARY_V2))
    if daily_df.empty:
        warnings.append(f"daily_summary_missing_or_empty={ADSB_DAILY_SUMMARY_V2}")

    timebin_df = _read_csv_with_date(TIMEBIN_SUMMARY_PATH)
    if timebin_df.empty:
        warnings.append(f"timebin_summary_missing_or_empty={TIMEBIN_SUMMARY_PATH}")

    mapping_df, map_warnings = build_phase_config_daily_mapping(daily_df=daily_df, timebin_df=timebin_df)
    warnings.extend(map_warnings)
    mapping_df.to_csv(MAPPING_CSV_PATH, index=False, encoding="utf-8-sig")
    _write_json_records(MAPPING_JSON_PATH, mapping_df)

    phase_timebin_df, agg_warnings = build_phase_timebin_summary(timebin_df=timebin_df, mapping_df=mapping_df)
    warnings.extend(agg_warnings)
    phase_timebin_df.to_csv(PHASE_TIMEBIN_CSV_PATH, index=False, encoding="utf-8-sig")
    _write_json_records(PHASE_TIMEBIN_JSON_PATH, phase_timebin_df)

    lines = [
        "Phase Time-bin Export Report",
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"daily_summary_path: {ADSB_DAILY_SUMMARY_V2}",
        f"timebin_summary_path: {TIMEBIN_SUMMARY_PATH}",
        f"mapping_csv: {MAPPING_CSV_PATH}",
        f"mapping_json: {MAPPING_JSON_PATH}",
        f"phase_timebin_csv: {PHASE_TIMEBIN_CSV_PATH}",
        f"phase_timebin_json: {PHASE_TIMEBIN_JSON_PATH}",
        f"mapping_rows: {len(mapping_df)}",
        f"phase_timebin_rows: {len(phase_timebin_df)}",
        "warnings:",
    ]
    if warnings:
        lines.extend([f"- {w}" for w in warnings])
    else:
        lines.append("- (none)")

    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    log.info(f"[OK] {MAPPING_CSV_PATH}")
    log.info(f"[OK] {PHASE_TIMEBIN_CSV_PATH}")
    log.info(f"[OK] {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
