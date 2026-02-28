# adsb_eval_pk_aggregator.py
# Purpose:
#  - Build adsb_daily_summary.csv for statistical modeling (robust & reproducible)
#  - Daily AUC is defined as SUM of n_used from dist_1m records (NOT max(n_total))
#  - Also outputs minutes_covered to justify day-quality filtering (avoid iloc[1:-1])
#
# Notes:
#  - dist_1m current: <project>/data/dist_1m.jsonl
#  - dist_1m archive (rotated): <project>/data/raw/past_log/*dist*.jsonl.till-*
#    (e.g., adsb_perf_dist.jsonl.till-202602100820)
#  - Daily boundary uses JST (Asia/Tokyo) by default.

import os
import sys
from pathlib import Path
import json
import glob
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


from arena.lib.paths import DATA_DIR, RAW_DIR, OUTPUT_DIR as OUT_ROOT

# --- Paths ---
BASE_DIR = str(DATA_DIR)
RAW_DIR = str(RAW_DIR)
PAST_LOG_DIR = os.path.join(RAW_DIR, "past_log")
TRAFFIC_DIR = os.path.join(BASE_DIR, "flight_data")
OUTPUT_DIR = str(OUT_ROOT)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "adsb_daily_summary.csv")

# External data
OPENSKY_FILE = os.path.join(TRAFFIC_DIR, "airport_movements.csv")  # expects columns: date, hnd_nrt_movements
FR24_CHART_FILE = os.path.join(BASE_DIR, "chart.csv")              # expects columns: DateTime, Tracked flights

# dist sources
DIST_CURRENT = os.path.join(BASE_DIR, "dist_1m.jsonl")
DIST_ARCHIVE_GLOB = os.path.join(PAST_LOG_DIR, "*dist*.jsonl.till-*")

# pos sources (optional)
POS_GLOB = os.path.join(BASE_DIR, "pos_*.jsonl")  # pos_YYYYMMDD.jsonl

# Timezone for daily cut
LOCAL_TZ = "Asia/Tokyo"  # JST
MAX_WORKERS = min(6, os.cpu_count() or 6)

_LOCAL_TZ_INFO = ZoneInfo(LOCAL_TZ) if ZoneInfo else None


def iter_jsonl_records(path: str):
    """Streaming JSONL reader. Yields dict per line."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _local_date_from_ts(ts):
    if _LOCAL_TZ_INFO:
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(_LOCAL_TZ_INFO).date()
    return pd.to_datetime(ts, unit="s", utc=True).tz_convert(LOCAL_TZ).date()


def _as_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _agg_dist_file(fp: str):
    per_day = {}
    for d in iter_jsonl_records(fp):
        src = d.get("src", "")
        if src and src != "dist_1m":
            continue
        if "ts" not in d or "n_used" not in d:
            continue

        ts = d.get("ts")
        if ts is None:
            continue

        day = _local_date_from_ts(ts)
        entry = per_day.setdefault(day, [0, 0, 0, 0, 0])
        entry[0] += _as_int(d.get("n_used"))
        entry[1] += 1
        entry[2] += _as_int(d.get("n_total"))
        entry[3] += _as_int(d.get("n_with_pos"))
        entry[4] += _as_int(d.get("n_fresh"))
    return per_day


def get_daily_auc_from_dist():
    """
    Build daily AUC from dist_1m records:
      - auc_n_used(day) = SUM(n_used per minute)
      - minutes_covered(day) = count(records per day)
      - Also keep daily sums for n_total/n_with_pos/n_fresh/n_used (optional for later modeling)
    """
    dist_files = []
    if os.path.exists(DIST_CURRENT):
        dist_files.append(DIST_CURRENT)
    dist_files.extend(glob.glob(DIST_ARCHIVE_GLOB))

    if not dist_files:
        return pd.DataFrame()

    merged = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for per_day in ex.map(_agg_dist_file, dist_files):
            for day, vals in per_day.items():
                entry = merged.setdefault(day, [0, 0, 0, 0, 0])
                entry[0] += vals[0]
                entry[1] += vals[1]
                entry[2] += vals[2]
                entry[3] += vals[3]
                entry[4] += vals[4]

    if not merged:
        return pd.DataFrame()

    daily = pd.DataFrame(
        [
            {
                "date": day,
                "auc_n_used": vals[0],
                "minutes_covered": vals[1],
                "sum_n_total": vals[2],
                "sum_n_with_pos": vals[3],
                "sum_n_fresh": vals[4],
            }
            for day, vals in merged.items()
        ]
    )

    # Keep types clean
    daily["auc_n_used"] = daily["auc_n_used"].astype("int64")
    daily["minutes_covered"] = daily["minutes_covered"].astype("int64")

    return daily


def get_daily_pos_stats():
    """
    Optional: derive route proxy and estimated overflights from pos_*.jsonl.
    - route_proxy_lat = mean(lat)
    - est_overflights = unique hex count satisfying altitude & speed thresholds
      (Thresholds are heuristic; keep as "proxy", not ground truth.)
    """
    pos_files = glob.glob(POS_GLOB)
    if not pos_files:
        return pd.DataFrame()

    def process_pos_file(pf: str):
        base = os.path.basename(pf)
        try:
            date_str = base.replace("pos_", "").replace(".jsonl", "")
            day = datetime.strptime(date_str, "%Y%m%d").date()
        except Exception:
            return None

        lats = []
        over_hex = set()
        any_seen = False

        for d in iter_jsonl_records(pf):
            any_seen = True
            lat = d.get("lat")
            if lat is not None:
                try:
                    lats.append(float(lat))
                except Exception:
                    pass

            try:
                alt = float(d.get("alt", np.nan))
                gs = float(d.get("gs", np.nan))
                hx = d.get("hex")
                if hx and alt > 24000 and gs > 400:
                    over_hex.add(hx)
            except Exception:
                pass

        if not any_seen:
            return None

        route_proxy_lat = float(np.mean(lats)) if lats else np.nan
        return {
            "date": day,
            "route_proxy_lat": route_proxy_lat,
            "est_overflights": int(len(over_hex)),
        }

    stats = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for result in ex.map(process_pos_file, pos_files):
            if result:
                stats.append(result)

    if not stats:
        return pd.DataFrame()

    out = pd.DataFrame(stats)
    return out


def merge_traffic(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Merge movements/traffic into summary.
    - Prefer OPENSKY_FILE if present (date, hnd_nrt_movements)
    - Optionally override with FR24 chart if present (Tracked flights)
    - Do NOT fill missing with arbitrary constant (like 1000). Keep missing + flag.
    """
    summary = summary.copy()
    summary["hnd_nrt_movements"] = np.nan

    # OpenSky
    if os.path.exists(OPENSKY_FILE):
        try:
            df_os = pd.read_csv(OPENSKY_FILE)
            df_os["date"] = pd.to_datetime(df_os["date"]).dt.date
            df_os = df_os[["date", "hnd_nrt_movements"]].copy()
            summary = pd.merge(summary, df_os, on="date", how="left", suffixes=("", "_os"))
            # If we ever had pre-existing, choose non-null; currently base is NaN anyway.
            summary["hnd_nrt_movements"] = summary["hnd_nrt_movements"].fillna(summary.get("hnd_nrt_movements_os"))
            if "hnd_nrt_movements_os" in summary.columns:
                summary.drop(columns=["hnd_nrt_movements_os"], inplace=True)
        except Exception:
            pass

    # FR24 chart (optional override)
    if os.path.exists(FR24_CHART_FILE):
        try:
            df_fr = pd.read_csv(FR24_CHART_FILE)
            # Expect DateTime + Tracked flights
            if "DateTime" in df_fr.columns and "Tracked flights" in df_fr.columns:
                df_fr["date"] = pd.to_datetime(df_fr["DateTime"]).dt.date
                df_fr = df_fr.groupby("date", as_index=False)["Tracked flights"].max()
                # Override where available
                summary = pd.merge(summary, df_fr, on="date", how="left")
                # Only override if not null
                summary["hnd_nrt_movements"] = summary["Tracked flights"].combine_first(summary["hnd_nrt_movements"])
                summary.drop(columns=["Tracked flights"], inplace=True)
        except Exception:
            pass

    # Missing flag (for analysis stage)
    summary["traffic_missing"] = summary["hnd_nrt_movements"].isna().astype(int)

    return summary


def main():
    print("🚀 ADS-B Eval PK Aggregator (Robust AUC from dist_1m): start")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Build daily AUC from dist_1m
    print("Step 1: Build daily AUC from dist_1m (SUM n_used) ...")
    summary = get_daily_auc_from_dist()
    if summary.empty:
        print("❌ dist_1m sources not found or empty. Cannot build summary.")
        print(f"   Expected: {DIST_CURRENT} and/or {DIST_ARCHIVE_GLOB}")
        return

    # Step 2: Merge optional pos-derived proxies
    print("Step 2: Merge optional pos_* proxies (route_proxy_lat, est_overflights) ...")
    pos_stats = get_daily_pos_stats()
    if not pos_stats.empty:
        summary = pd.merge(summary, pos_stats, on="date", how="left")
    else:
        summary["route_proxy_lat"] = np.nan
        summary["est_overflights"] = np.nan

    # Step 3: Merge traffic (movements)
    print("Step 3: Merge traffic (OpenSky/FR24 if available) ...")
    summary = merge_traffic(summary)

    # Step 4: Derive convenience columns
    summary["day_of_week"] = pd.to_datetime(summary["date"]).dt.strftime("%a")

    # Ensure numeric types
    for col in ["est_overflights"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    summary["route_proxy_lat"] = pd.to_numeric(summary["route_proxy_lat"], errors="coerce")
    summary["hnd_nrt_movements"] = pd.to_numeric(summary["hnd_nrt_movements"], errors="coerce")

    # Sorting
    summary = summary.sort_values("date").reset_index(drop=True)

    # Save
    summary.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Wrote: {OUTPUT_FILE}  rows={len(summary)}")
    print("Columns:", ", ".join(summary.columns))

    print("\nNext (analysis tip):")
    print("  - Use minutes_covered to filter incomplete days, e.g. minutes_covered >= 1380")
    print("  - For traffic outliers/missing, filter traffic_missing==0 and hnd_nrt_movements>=threshold")


if __name__ == "__main__":
    main()
