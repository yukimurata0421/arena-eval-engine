#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
adsb_opensky_comparison_eval.py module.
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import glob
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, kruskal
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from arena.lib.config import get_distance_bins_km, get_site_latlon
from arena.lib.stats_utils import bootstrap_mean_diff
from arena.lib.paths import DATA_DIR, OUTPUT_DIR, ensure_dir
from arena.lib.phase_config import get_config


# ════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════

SITE_LAT, SITE_LON = get_site_latlon()

DISTANCE_BINS_KM = get_distance_bins_km()
DISTANCE_BIN_LABELS = [
    f"{int(DISTANCE_BINS_KM[i])}-{int(DISTANCE_BINS_KM[i + 1])}"
    for i in range(len(DISTANCE_BINS_KM) - 1)
]

DEFAULT_OS_MIN_N_USED = 3
DEFAULT_OS_MAX_KM_MAX = 600.0
DEFAULT_OS_MIN_MINUTES_PER_DAY = 60

DEFAULT_LOCAL_MIN_MINUTES_MATCH = 60

# capture_ratio
DEFAULT_CR_CAP = 5.0


# ════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════

def normalize_path(p: str) -> str:
    """Convert Windows path to WSL path (only when running in WSL)."""
    if os.name == "nt":
        return p
    m = re.match(r"^([A-Za-z]):[/\\](.*)$", p)
    if not m:
        return p
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def ts_to_utc_minute(ts: float) -> int:
    return int(ts // 60)


def ts_to_date_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d")


def date_str_to_iso(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def classify_distance_bin(dist_km: float) -> Optional[str]:
    for i in range(len(DISTANCE_BINS_KM) - 1):
        if DISTANCE_BINS_KM[i] <= dist_km < DISTANCE_BINS_KM[i + 1]:
            return DISTANCE_BIN_LABELS[i]
    return None


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


# ════════════════════════════════════════════════
# Phase resolution
# ════════════════════════════════════════════════

def resolve_phase_for_date(date_iso: str, cfg) -> str:
    hw = cfg.hardware_at(date_iso)
    pretty = {"rtl-sdr": "RTL-SDR", "airspy_mini": "Airspy Mini", "airspy_mini_plus_cable": "Airspy+Cable"}
    return pretty.get(hw, hw)


def resolve_phase_detailed(date_iso: str, cfg) -> str:
    result_label = "Unknown"
    for e in cfg.events:
        if e.date <= date_iso:
            result_label = e.label
        else:
            break
    return result_label


# ════════════════════════════════════════════════
# ════════════════════════════════════════════════

@dataclass
class OpenSkyMinute:
    ts: float
    minute_key: int
    date_str: str
    n_total: int
    n_used: int
    km_n: int
    km_avg: float
    km_p50: float
    km_p75: float
    km_p90: float
    km_p95: float
    km_max: float


def load_opensky_data(
    opensky_dir: str,
    pattern: str = "dist_1m.jsonl",
    min_n_used: int = DEFAULT_OS_MIN_N_USED,
    max_km_max: float = DEFAULT_OS_MAX_KM_MAX,
) -> Tuple[List[OpenSkyMinute], Dict[str, Any]]:
    """
    Load OpenSky dist_1m.jsonl and apply quality filters.
    Returns: (filtered_records, quality_report_dict)
    """
    paths = sorted(glob.glob(os.path.join(opensky_dir, pattern)))
    if not paths:
        single = os.path.join(opensky_dir, "dist_1m.jsonl")
        if os.path.exists(single):
            paths = [single]

    raw_count = 0
    results = []
    rejected_low_n = 0
    rejected_high_km = 0

    for p in paths:
        for rec in iter_jsonl(p):
            raw_count += 1
            ts = rec.get("ts")
            if ts is None:
                continue

            km = rec.get("km", {})
            n_used = int(rec.get("n_used", 0) or rec.get("n_fresh", 0) or 0)
            km_max_val = float(km.get("max", 0))

            if n_used < min_n_used:
                rejected_low_n += 1
                continue
            if km_max_val > max_km_max:
                rejected_high_km += 1
                continue

            osm = OpenSkyMinute(
                ts=float(ts),
                minute_key=ts_to_utc_minute(float(ts)),
                date_str=ts_to_date_str(float(ts)),
                n_total=int(rec.get("n_total", 0)),
                n_used=n_used,
                km_n=int(km.get("n", 0)),
                km_avg=float(km.get("avg", 0)),
                km_p50=float(km.get("p50", 0)),
                km_p75=float(km.get("p75", 0)),
                km_p90=float(km.get("p90", 0)),
                km_p95=float(km.get("p95", 0)),
                km_max=km_max_val,
            )
            results.append(osm)

    quality_report = {
        "raw_records": raw_count,
        "accepted": len(results),
        "rejected_low_n_used": rejected_low_n,
        "rejected_high_km_max": rejected_high_km,
        "min_n_used_threshold": min_n_used,
        "max_km_max_threshold": max_km_max,
    }

    return results, quality_report


# ════════════════════════════════════════════════
# ════════════════════════════════════════════════

@dataclass
class LocalMinuteSummary:
    minute_key: int
    date_str: str
    n_unique: int
    km_max: float
    bin_counts: Dict[str, int]


def get_existing_local_dates(local_dir: str, pattern: str = "pos_*.jsonl") -> set:
    """Return date set (YYYYMMDD) where local pos files exist."""
    dates = set()
    paths = glob.glob(os.path.join(local_dir, pattern))
    for p in paths:
        m = re.search(r"pos_(\d{8})\.jsonl$", os.path.basename(p))
        if m:
            dates.add(m.group(1))
    return dates


def load_local_data(
    local_dir: str,
    pattern: str = "pos_*.jsonl",
    target_dates: Optional[set] = None,
    site_latlon: Tuple[float, float] = (SITE_LAT, SITE_LON),
) -> Dict[int, LocalMinuteSummary]:
    paths = sorted(glob.glob(os.path.join(local_dir, pattern)))

    minute_hex_dist: Dict[int, Dict[str, float]] = defaultdict(dict)
    minute_dates: Dict[int, str] = {}

    for p in paths:
        date_from_fn = None
        m = re.search(r"pos_(\d{8})\.jsonl$", os.path.basename(p))
        if m:
            date_from_fn = m.group(1)

        if target_dates and date_from_fn and date_from_fn not in target_dates:
            continue

        print(f"  [local] loading {os.path.basename(p)} ...", flush=True)

        for rec in iter_jsonl(p):
            if rec.get("type") != "pos":
                continue
            if int(rec.get("schema_ver", 0) or 0) != 1:
                continue

            ts = rec.get("ts")
            hx = rec.get("hex")
            lat = rec.get("lat")
            lon = rec.get("lon")

            if ts is None or hx is None or lat is None or lon is None:
                continue

            try:
                ts_f = float(ts)
                lat_f = float(lat)
                lon_f = float(lon)
            except (ValueError, TypeError):
                continue

            mk = ts_to_utc_minute(ts_f)
            dist_km = haversine_km(site_latlon[0], site_latlon[1], lat_f, lon_f)

            if hx not in minute_hex_dist[mk] or dist_km > minute_hex_dist[mk][hx]:
                minute_hex_dist[mk][hx] = dist_km

            if mk not in minute_dates:
                minute_dates[mk] = date_from_fn or ts_to_date_str(ts_f)

    result: Dict[int, LocalMinuteSummary] = {}
    for mk, hex_dists in minute_hex_dist.items():
        bin_counts = {lab: 0 for lab in DISTANCE_BIN_LABELS}
        km_max = 0.0
        for hx, dist in hex_dists.items():
            blab = classify_distance_bin(dist)
            if blab:
                bin_counts[blab] += 1
            if dist > km_max:
                km_max = dist

        result[mk] = LocalMinuteSummary(
            minute_key=mk,
            date_str=minute_dates.get(mk, ""),
            n_unique=len(hex_dists),
            km_max=km_max,
            bin_counts=bin_counts,
        )

    return result


# ════════════════════════════════════════════════
# ════════════════════════════════════════════════

def estimate_opensky_bin_counts(osm: OpenSkyMinute) -> Dict[str, float]:
    n = osm.km_n
    if n <= 0:
        return {lab: 0.0 for lab in DISTANCE_BIN_LABELS}

    cdf_points = [
        (0.0, 0.0),
        (osm.km_p50, 0.50),
        (osm.km_p75, 0.75),
        (osm.km_p90, 0.90),
        (osm.km_p95, 0.95),
        (osm.km_max, 1.00),
    ]

    def cdf_at(km: float) -> float:
        if km <= 0:
            return 0.0
        if km >= osm.km_max:
            return 1.0
        for i in range(len(cdf_points) - 1):
            d0, f0 = cdf_points[i]
            d1, f1 = cdf_points[i + 1]
            if d0 <= km <= d1:
                if d1 == d0:
                    return f1
                ratio = (km - d0) / (d1 - d0)
                return f0 + ratio * (f1 - f0)
        return 1.0

    result = {}
    for i in range(len(DISTANCE_BINS_KM) - 1):
        lo = float(DISTANCE_BINS_KM[i])
        hi = float(DISTANCE_BINS_KM[i + 1])
        if hi >= 9999:
            hi = max(osm.km_max + 1, 300.0)
        frac = cdf_at(hi) - cdf_at(lo)
        result[DISTANCE_BIN_LABELS[i]] = max(0.0, frac * n)

    return result


# ════════════════════════════════════════════════
# ════════════════════════════════════════════════

@dataclass
class MergedMinute:
    minute_key: int
    date_str: str
    date_iso: str
    phase: str
    phase_detail: str
    os_n_total: int
    os_n_used: int
    os_km_max: float
    os_km_avg: float
    os_bin_counts: Dict[str, float]
    local_n_unique: int
    local_km_max: float
    local_bin_counts: Dict[str, int]
    capture_ratio: float
    capture_by_bin: Dict[str, float]


def merge_data(
    opensky_data: List[OpenSkyMinute],
    local_data: Dict[int, LocalMinuteSummary],
    cfg,
    cr_cap: float = DEFAULT_CR_CAP,
) -> List[MergedMinute]:
    results = []

    for osm in opensky_data:
        mk = osm.minute_key
        lm = local_data.get(mk)

        date_iso = date_str_to_iso(osm.date_str)
        phase = resolve_phase_for_date(date_iso, cfg)
        phase_detail = resolve_phase_detailed(date_iso, cfg)

        os_bins = estimate_opensky_bin_counts(osm)

        if lm is None:
            local_n = 0
            local_km_max = 0.0
            local_bins = {lab: 0 for lab in DISTANCE_BIN_LABELS}
        else:
            local_n = lm.n_unique
            local_km_max = lm.km_max
            local_bins = lm.bin_counts

        os_count = osm.n_used if osm.n_used > 0 else osm.n_total
        cr = local_n / os_count if os_count > 0 else np.nan

        if not np.isnan(cr) and cr > cr_cap:
            cr = cr_cap

        capture_by_bin = {}
        for lab in DISTANCE_BIN_LABELS:
            os_b = os_bins.get(lab, 0.0)
            lc_b = float(local_bins.get(lab, 0))
            cb = lc_b / os_b if os_b > 0.5 else np.nan
            if not np.isnan(cb) and cb > cr_cap:
                cb = cr_cap
            capture_by_bin[lab] = cb

        results.append(MergedMinute(
            minute_key=mk,
            date_str=osm.date_str,
            date_iso=date_iso,
            phase=phase,
            phase_detail=phase_detail,
            os_n_total=osm.n_total,
            os_n_used=osm.n_used,
            os_km_max=osm.km_max,
            os_km_avg=osm.km_avg,
            os_bin_counts=os_bins,
            local_n_unique=local_n,
            local_km_max=local_km_max,
            local_bin_counts=local_bins,
            capture_ratio=cr,
            capture_by_bin=capture_by_bin,
        ))

    return results


# ════════════════════════════════════════════════
# ════════════════════════════════════════════════

def merged_to_dataframe(merged: List[MergedMinute]) -> pd.DataFrame:
    rows = []
    for mm in merged:
        row = {
            "minute_key": mm.minute_key,
            "date": mm.date_str,
            "date_iso": mm.date_iso,
            "phase": mm.phase,
            "phase_detail": mm.phase_detail,
            "os_n_total": mm.os_n_total,
            "os_n_used": mm.os_n_used,
            "os_km_max": mm.os_km_max,
            "os_km_avg": mm.os_km_avg,
            "local_n_unique": mm.local_n_unique,
            "local_km_max": mm.local_km_max,
            "capture_ratio": mm.capture_ratio,
        }
        for lab in DISTANCE_BIN_LABELS:
            row[f"os_bin_{lab}"] = mm.os_bin_counts.get(lab, 0.0)
            row[f"local_bin_{lab}"] = mm.local_bin_counts.get(lab, 0)
            row[f"capture_bin_{lab}"] = mm.capture_by_bin.get(lab, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def make_daily_summary(
    df: pd.DataFrame,
    local_dates_exist: set,
    os_min_minutes: int = DEFAULT_OS_MIN_MINUTES_PER_DAY,
    local_min_minutes: int = DEFAULT_LOCAL_MIN_MINUTES_MATCH,
) -> pd.DataFrame:
    """
    Create daily summary and set use_for_stats flag.

    Exclusion criteria:
      1. days without pos files (not in local_dates_exist)
      2. OpenSky valid minutes < os_min_minutes
      3. Local valid minutes < local_min_minutes
    """
    agg_dict = {
        "minute_key": "count",
        "os_n_used": "mean",
        "local_n_unique": "mean",
        "capture_ratio": "median",
        "os_km_max": "max",
        "local_km_max": "max",
    }
    rename_dict = {
        "minute_key": "n_minutes",
        "os_n_used": "os_mean_n_used",
        "local_n_unique": "local_mean_n_unique",
        "capture_ratio": "median_capture_ratio",
        "os_km_max": "os_max_km",
        "local_km_max": "local_max_km",
    }

    for lab in DISTANCE_BIN_LABELS:
        agg_dict[f"capture_bin_{lab}"] = "median"
        rename_dict[f"capture_bin_{lab}"] = f"median_capture_{lab}"

    daily = df.groupby(["date", "date_iso", "phase", "phase_detail"]).agg(agg_dict)
    daily = daily.rename(columns=rename_dict).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)

    local_nonzero = df[df["local_n_unique"] > 0].groupby("date").size().reset_index(name="local_nonzero_minutes")
    daily = daily.merge(local_nonzero, on="date", how="left")
    daily["local_nonzero_minutes"] = daily["local_nonzero_minutes"].fillna(0).astype(int)

    has_pos_file = daily["date"].isin(local_dates_exist)
    has_enough_os = daily["n_minutes"] >= os_min_minutes
    has_enough_local = daily["local_nonzero_minutes"] >= local_min_minutes

    daily["pos_file_exists"] = has_pos_file
    daily["use_for_stats"] = has_pos_file & has_enough_os & has_enough_local

    reasons = []
    for _, r in daily.iterrows():
        if r["use_for_stats"]:
            reasons.append("")
        elif not r["pos_file_exists"]:
            reasons.append("no_pos_file")
        elif r["n_minutes"] < os_min_minutes:
            reasons.append(f"os_minutes<{os_min_minutes}")
        elif r["local_nonzero_minutes"] < local_min_minutes:
            reasons.append(f"local_nonzero<{local_min_minutes}")
        else:
            reasons.append("unknown")
    daily["skip_reason"] = reasons

    return daily


# ════════════════════════════════════════════════
# ════════════════════════════════════════════════

def run_statistics(df_all: pd.DataFrame, daily: pd.DataFrame) -> List[str]:
    """
    Stats use only days where use_for_stats=True.
    """
    lines = []
    sep = "=" * 70

    used_dates = set(daily.loc[daily["use_for_stats"], "date"].values)
    df = df_all[df_all["date"].isin(used_dates)].copy()
    daily_used = daily[daily["use_for_stats"]].copy()

    phases = sorted(df["phase"].unique())

    lines.append(sep)
    lines.append("DATA QUALITY SUMMARY")
    lines.append(sep)
    lines.append(f"Total days in data:       {len(daily)}")
    lines.append(f"Days with pos file:       {int(daily['pos_file_exists'].sum())}")
    lines.append(f"Days used for stats:      {int(daily['use_for_stats'].sum())}")
    lines.append(f"Days skipped:             {int((~daily['use_for_stats']).sum())}")
    lines.append(f"Total minutes (all):      {len(df_all)}")
    lines.append(f"Total minutes (used):     {len(df)}")
    lines.append(f"Phases in used data:      {phases}")
    lines.append("")

    skip_summary = daily[~daily["use_for_stats"]].groupby("skip_reason").size()
    for reason, cnt in skip_summary.items():
        lines.append(f"  skip: {reason} = {cnt} days")
    lines.append("")

    if len(df) == 0:
        lines.append("[ERROR] No valid data after filtering. Cannot compute statistics.")
        return lines

    # ──────────────────────────────────────────
    # ──────────────────────────────────────────
    lines.append(sep)
    lines.append("1. Descriptive Statistics (used days only)")
    lines.append(sep)
    lines.append("")

    for ph in phases:
        d = df[df["phase"] == ph]
        n_days = int(daily_used[daily_used["phase"] == ph].shape[0])
        lines.append(f"-- Phase: {ph} (n_days={n_days}, n_minutes={len(d)}) --")
        lines.append(f"  OpenSky  n_used:  mean={d['os_n_used'].mean():.1f}  median={d['os_n_used'].median():.1f}")
        lines.append(f"  Local    n_unique: mean={d['local_n_unique'].mean():.1f}  median={d['local_n_unique'].median():.1f}")
        cr = d["capture_ratio"].dropna()
        if len(cr) > 0:
            lines.append(f"  Capture ratio:     mean={cr.mean():.4f}  median={cr.median():.4f}  std={cr.std():.4f}")
        lines.append(f"  Reach: os_km_max={d['os_km_max'].max():.1f}  local_km_max={d['local_km_max'].max():.1f}")

        for lab in DISTANCE_BIN_LABELS:
            vals = d[f"capture_bin_{lab}"].dropna()
            if len(vals) > 0:
                lines.append(f"  Capture {lab:>8s} km:  mean={vals.mean():.4f}  median={vals.median():.4f}  n={len(vals)}")
        lines.append("")

    # ──────────────────────────────────────────
    # 2. Mann-Whitney U (capture_ratio)
    # ──────────────────────────────────────────
    lines.append(sep)
    lines.append("2. Phase comparison: Mann-Whitney U (capture_ratio, minute-level)")
    lines.append(sep)
    lines.append("")

    if len(phases) >= 2:
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                ph_a, ph_b = phases[i], phases[j]
                a = df.loc[df["phase"] == ph_a, "capture_ratio"].dropna().values
                b = df.loc[df["phase"] == ph_b, "capture_ratio"].dropna().values
                if len(a) >= 5 and len(b) >= 5:
                    u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
                    md, lo, hi = bootstrap_mean_diff(a, b, n=20000, seed=42)
                    lines.append(f"  {ph_a} (n={len(a)}) vs {ph_b} (n={len(b)})")
                    lines.append(f"    MWU: U={u_stat:.1f}  p={u_p:.6g}")
                    lines.append(f"    Bootstrap delta_mean({ph_b}-{ph_a}): {md:.4f}  95%CI[{lo:.4f}, {hi:.4f}]")
                    lines.append("")
    else:
        lines.append("  Only 1 phase in valid data — no comparison possible.")
        lines.append("")

    # ──────────────────────────────────────────
    # ──────────────────────────────────────────
    lines.append(sep)
    lines.append("3. Distance-bin phase comparison (Mann-Whitney U on capture_bin)")
    lines.append(sep)
    lines.append("")

    if len(phases) >= 2:
        for lab in DISTANCE_BIN_LABELS:
            col = f"capture_bin_{lab}"
            lines.append(f"-- {lab} km --")
            for i in range(len(phases)):
                for j in range(i + 1, len(phases)):
                    ph_a, ph_b = phases[i], phases[j]
                    a = df.loc[df["phase"] == ph_a, col].dropna().values
                    b = df.loc[df["phase"] == ph_b, col].dropna().values
                    if len(a) >= 5 and len(b) >= 5:
                        u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
                        md, lo, hi = bootstrap_mean_diff(a, b, n=10000, seed=42)
                        lines.append(f"  {ph_a} vs {ph_b}: MWU p={u_p:.6g}  delta={md:.4f} [{lo:.4f},{hi:.4f}]")
                    else:
                        lines.append(f"  {ph_a} vs {ph_b}: insufficient data (n={len(a)}/{len(b)})")
            lines.append("")

    # ──────────────────────────────────────────
    # ──────────────────────────────────────────
    lines.append(sep)
    lines.append("4. Daily-level comparison (MWU on daily median_capture_ratio)")
    lines.append(sep)
    lines.append("")

    if len(phases) >= 2 and len(daily_used) >= 4:
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                ph_a, ph_b = phases[i], phases[j]
                a = daily_used.loc[daily_used["phase"] == ph_a, "median_capture_ratio"].dropna().values
                b = daily_used.loc[daily_used["phase"] == ph_b, "median_capture_ratio"].dropna().values
                if len(a) >= 3 and len(b) >= 3:
                    u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
                    md, lo, hi = bootstrap_mean_diff(a, b, n=20000, seed=42)
                    lines.append(f"  {ph_a} (n_days={len(a)}, mean={np.mean(a):.4f})")
                    lines.append(f"  vs {ph_b} (n_days={len(b)}, mean={np.mean(b):.4f})")
                    lines.append(f"    MWU: U={u_stat:.1f}  p={u_p:.6g}")
                    lines.append(f"    Bootstrap delta: {md:.4f} [{lo:.4f},{hi:.4f}]")
                    lines.append("")
                else:
                    lines.append(f"  {ph_a} (n={len(a)}) vs {ph_b} (n={len(b)}): need >=3 each")
                    lines.append("")
    else:
        lines.append("  Insufficient data for daily comparison.")
        lines.append("")

    # ──────────────────────────────────────────
    # ──────────────────────────────────────────
    lines.append(sep)
    lines.append("5. Negative Binomial GLM: local_n_unique ~ phase + offset(log(os_n_used))")
    lines.append(sep)
    lines.append("")

    try:
        d_glm = df[df["os_n_used"] > 0].copy()
        d_glm["log_os"] = np.log(d_glm["os_n_used"].astype(float))

        if len(d_glm["phase"].unique()) >= 2:
            formula = "local_n_unique ~ C(phase)"
        else:
            formula = "local_n_unique ~ 1"

        model = smf.glm(
            formula=formula,
            data=d_glm,
            family=sm.families.NegativeBinomial(alpha=1.0),
            offset=d_glm["log_os"],
        )
        res = model.fit()
        lines.append(res.summary().as_text())
    except Exception as e:
        lines.append(f"  [ERROR] NB-GLM failed: {repr(e)}")
    lines.append("")

    # ──────────────────────────────────────────
    # ──────────────────────────────────────────
    lines.append(sep)
    lines.append("6. Distance-bin NB-GLM: local_bin ~ phase + offset(log(os_bin))")
    lines.append(sep)
    lines.append("")

    for lab in DISTANCE_BIN_LABELS:
        lines.append(f"-- {lab} km --")
        try:
            d_bin = df[["phase", f"local_bin_{lab}", f"os_bin_{lab}"]].copy()
            d_bin.columns = ["phase", "local_count", "os_count"]
            d_bin = d_bin[d_bin["os_count"] > 0.5].copy()
            d_bin["local_count"] = d_bin["local_count"].astype(int)
            d_bin["log_os"] = np.log(d_bin["os_count"].astype(float))

            if len(d_bin) < 10:
                lines.append("  insufficient data")
                lines.append("")
                continue

            if len(d_bin["phase"].unique()) >= 2:
                formula = "local_count ~ C(phase)"
            else:
                formula = "local_count ~ 1"

            model = smf.glm(
                formula=formula,
                data=d_bin,
                family=sm.families.NegativeBinomial(alpha=1.0),
                offset=d_bin["log_os"],
            )
            res = model.fit()
            lines.append(res.summary().as_text())
        except Exception as e:
            lines.append(f"  [ERROR] {repr(e)}")
        lines.append("")

    # ──────────────────────────────────────────
    # 7. Kruskal-Wallis (3+)
    # ──────────────────────────────────────────
    if len(phases) >= 3:
        lines.append(sep)
        lines.append("7. Kruskal-Wallis test (capture_ratio across all phases)")
        lines.append(sep)
        lines.append("")

        groups = []
        group_names = []
        for ph in phases:
            vals = df.loc[df["phase"] == ph, "capture_ratio"].dropna().values
            if len(vals) >= 3:
                groups.append(vals)
                group_names.append(f"{ph}(n={len(vals)})")

        if len(groups) >= 3:
            h_stat, h_p = kruskal(*groups)
            lines.append(f"  Groups: {', '.join(group_names)}")
            lines.append(f"  H={h_stat:.4f}  p={h_p:.6g}")
        elif len(groups) >= 2:
            lines.append(f"  Only {len(groups)} groups with n>=3 (need 3+). Using 2-group MWU instead.")
        else:
            lines.append("  Not enough groups with n>=3")
        lines.append("")

    return lines


# ════════════════════════════════════════════════
# ════════════════════════════════════════════════

def _phase_colors() -> Dict[str, str]:
    return {
        "RTL-SDR": "#7f8c8d",
        "Airspy Mini": "#e74c3c",
        "Airspy+Cable": "#2980b9",
    }


def plot_capture_ratio_by_phase(df: pd.DataFrame, out_path: str) -> None:
    phases = sorted(df["phase"].unique())
    data = [df.loc[df["phase"] == ph, "capture_ratio"].dropna().values for ph in phases]
    colors = _phase_colors()

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=phases, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors.get(phases[i], "#cccccc"))
        patch.set_alpha(0.6)

    ax.set_title("Capture Ratio by Phase (Local / OpenSky)\n[valid days only, outliers hidden]")
    ax.set_ylabel("capture_ratio")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1.0 (perfect)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_capture_by_distance_bin(df: pd.DataFrame, out_path: str) -> None:
    phases = sorted(df["phase"].unique())
    n_phases = len(phases)
    n_bins = len(DISTANCE_BIN_LABELS)
    colors = _phase_colors()

    medians = np.zeros((n_phases, n_bins))
    q25 = np.zeros((n_phases, n_bins))
    q75 = np.zeros((n_phases, n_bins))

    for i, ph in enumerate(phases):
        d = df[df["phase"] == ph]
        for j, lab in enumerate(DISTANCE_BIN_LABELS):
            vals = d[f"capture_bin_{lab}"].dropna().values
            if len(vals) > 0:
                medians[i, j] = np.median(vals)
                q25[i, j] = np.quantile(vals, 0.25)
                q75[i, j] = np.quantile(vals, 0.75)

    x = np.arange(n_bins)
    width = 0.8 / n_phases

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, ph in enumerate(phases):
        offset = (i - n_phases / 2 + 0.5) * width
        yerr_lo = medians[i] - q25[i]
        yerr_hi = q75[i] - medians[i]
        ax.bar(x + offset, medians[i], width,
               yerr=[yerr_lo, yerr_hi],
               label=ph, color=colors.get(ph, "#cccccc"), alpha=0.7, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} km" for lab in DISTANCE_BIN_LABELS])
    ax.set_ylabel("Median capture ratio (IQR)")
    ax.set_title("Capture Ratio by Distance Bin x Phase\n[valid days only]")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_daily_trend_compact(daily_used: pd.DataFrame, out_path: str) -> None:
    """Pack only valid days (x-axis indices, labels are dates)."""
    d = daily_used.sort_values("date").reset_index(drop=True)
    x = np.arange(len(d))
    phases = sorted(d["phase"].unique())
    colors = _phase_colors()

    fig, ax = plt.subplots(figsize=(12, 5))
    for ph in phases:
        mask = d["phase"] == ph
        ax.plot(x[mask], d.loc[mask, "median_capture_ratio"].values, "o-",
                label=ph, color=colors.get(ph, "#2980b9"), alpha=0.8, markersize=5)

    ax.set_title("Daily Median Capture Ratio (compact; valid days only)")
    ax.set_ylabel("median_capture_ratio")
    ax.set_xlabel("Date (packed)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    step = max(1, len(d) // 15)
    tick_idx = list(range(0, len(d), step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([d.loc[i, "date_iso"] for i in tick_idx], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_distance_heatmap(df: pd.DataFrame, out_path: str) -> None:
    phases = sorted(df["phase"].unique())
    matrix = np.zeros((len(phases), len(DISTANCE_BIN_LABELS)))

    for i, ph in enumerate(phases):
        d = df[df["phase"] == ph]
        for j, lab in enumerate(DISTANCE_BIN_LABELS):
            vals = d[f"capture_bin_{lab}"].dropna()
            matrix[i, j] = vals.median() if len(vals) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, max(3, len(phases) * 1.2)))
    vmax = min(float(matrix.max()) * 1.2, 3.0) if matrix.max() > 0 else 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(DISTANCE_BIN_LABELS)))
    ax.set_xticklabels([f"{l} km" for l in DISTANCE_BIN_LABELS])
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels(phases)
    ax.set_title("Median Capture Ratio: Phase x Distance Bin\n[valid days only]")

    for i in range(len(phases)):
        for j in range(len(DISTANCE_BIN_LABELS)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="black" if matrix[i, j] > 0.4 else "white")

    fig.colorbar(im, ax=ax, label="capture ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_daily_bin_trend_compact(daily_used: pd.DataFrame, out_path: str) -> None:
    """Daily capture ratio trend by distance bin (compact)."""
    d = daily_used.sort_values("date").reset_index(drop=True)
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.viridis
    for i, lab in enumerate(DISTANCE_BIN_LABELS):
        col = f"median_capture_{lab}"
        if col in d.columns:
            ax.plot(x, d[col].values, "o-", label=f"{lab} km",
                    color=cmap(i / len(DISTANCE_BIN_LABELS)), alpha=0.7, markersize=3)

    ax.set_title("Daily Median Capture by Distance Bin (compact)")
    ax.set_ylabel("median capture ratio")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    step = max(1, len(d) // 15)
    tick_idx = list(range(0, len(d), step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([d.loc[i, "date_iso"] for i in tick_idx], rotation=45, ha="right")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="OpenSky vs Local ADS-B Comparison Evaluator v2")

    ap.add_argument("--opensky-dir", default=str(DATA_DIR),
                     help="OpenSky dist_1m.jsonl directory")
    ap.add_argument("--opensky-pattern", default="dist_1m.jsonl",
                     help="OpenSky file pattern")
    ap.add_argument("--local-dir", default=str(DATA_DIR / "plao_pos"),
                     help="Local pos_YYYYMMDD.jsonl directory")
    ap.add_argument("--local-pattern", default="pos_*.jsonl",
                     help="Local file pattern")

    ap.add_argument("--output-root", default=None)
    ap.add_argument("--out-subdir", default="opensky_comparison")

    ap.add_argument("--site-lat", type=float, default=SITE_LAT)
    ap.add_argument("--site-lon", type=float, default=SITE_LON)

    ap.add_argument("--phase-config", default=None)
    ap.add_argument("--plots", action="store_true")

    ap.add_argument("--os-min-n-used", type=int, default=DEFAULT_OS_MIN_N_USED,
                     help=f"OpenSky min n_used per minute (default: {DEFAULT_OS_MIN_N_USED})")
    ap.add_argument("--os-max-km-max", type=float, default=DEFAULT_OS_MAX_KM_MAX,
                     help=f"OpenSky max km_max threshold (default: {DEFAULT_OS_MAX_KM_MAX})")
    ap.add_argument("--os-min-minutes", type=int, default=DEFAULT_OS_MIN_MINUTES_PER_DAY,
                     help=f"Min OpenSky minutes per day (default: {DEFAULT_OS_MIN_MINUTES_PER_DAY})")
    ap.add_argument("--local-min-minutes", type=int, default=DEFAULT_LOCAL_MIN_MINUTES_MATCH,
                     help=f"Min local nonzero minutes per day (default: {DEFAULT_LOCAL_MIN_MINUTES_MATCH})")
    ap.add_argument("--cr-cap", type=float, default=DEFAULT_CR_CAP,
                     help=f"Capture ratio upper cap (default: {DEFAULT_CR_CAP})")

    args = ap.parse_args()

    args.opensky_dir = normalize_path(args.opensky_dir)
    args.local_dir = normalize_path(args.local_dir)

    if args.output_root:
        output_root = Path(normalize_path(args.output_root))
    else:
        output_root = OUTPUT_DIR

    out_dir = str(output_root / args.out_subdir)
    ensure_dir(out_dir)

    cfg = get_config(args.phase_config)

    site_lat = args.site_lat
    site_lon = args.site_lon

    print("=" * 70)
    print("OpenSky vs Local ADS-B Comparison Evaluator v2")
    print("=" * 70)
    print(f"  OpenSky dir:     {args.opensky_dir}")
    print(f"  Local dir:       {args.local_dir}")
    print(f"  Output:          {out_dir}")
    print(f"  Site:            ({site_lat:.6f}, {site_lon:.6f})")
    print(f"  Distance bins:   {DISTANCE_BIN_LABELS}")
    print(f"  Phase config:    {cfg.config_path}")
    print(f"  Quality filters:")
    print(f"    os_min_n_used:       {args.os_min_n_used}")
    print(f"    os_max_km_max:       {args.os_max_km_max}")
    print(f"    os_min_minutes/day:  {args.os_min_minutes}")
    print(f"    local_min_minutes:   {args.local_min_minutes}")
    print(f"    cr_cap:              {args.cr_cap}")
    print("")

    print("[1/6] Loading OpenSky data (with quality filter) ...")
    opensky_data, os_quality = load_opensky_data(
        args.opensky_dir, args.opensky_pattern,
        min_n_used=args.os_min_n_used,
        max_km_max=args.os_max_km_max,
    )
    if not opensky_data:
        print("  ERROR: No OpenSky data found!")
        sys.exit(1)
    print(f"  Raw records:          {os_quality['raw_records']}")
    print(f"  Accepted:             {os_quality['accepted']}")
    print(f"  Rejected (low n):     {os_quality['rejected_low_n_used']}")
    print(f"  Rejected (high km):   {os_quality['rejected_high_km_max']}")

    os_dates = set(osm.date_str for osm in opensky_data)
    print(f"  Dates: {len(os_dates)} days ({min(os_dates)} ~ {max(os_dates)})")

    print("\n[2/6] Scanning local pos files ...")
    local_dates_exist = get_existing_local_dates(args.local_dir, args.local_pattern)
    overlap = os_dates & local_dates_exist
    print(f"  Local pos files found: {len(local_dates_exist)}")
    print(f"  Overlap with OpenSky:  {len(overlap)}")

    if not overlap:
        print("  WARNING: No overlapping dates. All days will be excluded from stats.")

    print("\n[3/6] Loading local ADS-B data ...")
    local_data = load_local_data(
        args.local_dir, args.local_pattern,
        target_dates=overlap,
        site_latlon=(site_lat, site_lon),
    )
    print(f"  Loaded {len(local_data)} minute summaries")

    print("\n[4/6] Merging OpenSky + Local ...")
    merged = merge_data(opensky_data, local_data, cfg, cr_cap=args.cr_cap)
    df = merged_to_dataframe(merged)
    print(f"  Merged records: {len(df)}")
    print(f"  Phases found:   {sorted(df['phase'].unique())}")

    matched = df["local_n_unique"] > 0
    print(f"  Minutes with local data: {matched.sum()} / {len(df)} ({matched.mean()*100:.1f}%)")

    minutely_csv = os.path.join(out_dir, "opensky_local_minutely_merged.csv")
    df.to_csv(minutely_csv, index=False, encoding="utf-8-sig")
    print(f"  Wrote: {minutely_csv}")

    print("\n[5/6] Building daily summary (with quality flags) ...")
    daily = make_daily_summary(
        df, local_dates_exist,
        os_min_minutes=args.os_min_minutes,
        local_min_minutes=args.local_min_minutes,
    )
    daily_csv = os.path.join(out_dir, "opensky_comparison_daily_summary.csv")
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    print(f"  Wrote: {daily_csv}")

    n_used = int(daily["use_for_stats"].sum())
    n_skip = int((~daily["use_for_stats"]).sum())
    print(f"  Total days: {len(daily)}  |  Used: {n_used}  |  Skipped: {n_skip}")

    df_skip = daily[~daily["use_for_stats"]].copy()
    if len(df_skip) > 0:
        skip_csv = os.path.join(out_dir, "opensky_skipped_days.csv")
        df_skip.to_csv(skip_csv, index=False, encoding="utf-8-sig")
        print(f"  Wrote: {skip_csv}")

    print("\n[6/6] Running statistical analyses ...")
    report_lines = []
    report_lines.append("OpenSky vs Local ADS-B Comparison Report v2")
    report_lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"OpenSky dir: {args.opensky_dir}")
    report_lines.append(f"Local dir: {args.local_dir}")
    report_lines.append(f"Site: ({site_lat}, {site_lon})")
    report_lines.append("")
    report_lines.append("Quality thresholds:")
    report_lines.append(f"  os_min_n_used:     {args.os_min_n_used}")
    report_lines.append(f"  os_max_km_max:     {args.os_max_km_max}")
    report_lines.append(f"  os_min_minutes:    {args.os_min_minutes}")
    report_lines.append(f"  local_min_minutes: {args.local_min_minutes}")
    report_lines.append(f"  cr_cap:            {args.cr_cap}")
    report_lines.append("")
    report_lines.append("OpenSky data quality:")
    for k, v in os_quality.items():
        report_lines.append(f"  {k}: {v}")
    report_lines.append("")
    report_lines.append(f"Total minute records: {len(df)}")
    report_lines.append(f"Minutes with local data: {matched.sum()}")
    report_lines.append(f"Days total: {len(daily)}  used: {n_used}  skipped: {n_skip}")
    report_lines.append(f"Phases (all): {sorted(df['phase'].unique())}")

    used_dates = set(daily.loc[daily["use_for_stats"], "date"].values)
    df_used = df[df["date"].isin(used_dates)]
    report_lines.append(f"Phases (used): {sorted(df_used['phase'].unique())}")
    report_lines.append(f"Distance bins (km): {DISTANCE_BIN_LABELS}")
    report_lines.append("")

    stats_lines = run_statistics(df, daily)
    report_lines.extend(stats_lines)

    report_path = os.path.join(out_dir, "opensky_comparison_stats_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Wrote: {report_path}")

    if args.plots:
        daily_used = daily[daily["use_for_stats"]].copy()
        used_dates_set = set(daily_used["date"].values)
        df_for_plot = df[df["date"].isin(used_dates_set)].copy()

        print("\n  Generating plots (valid days only, compact) ...")
        if len(df_for_plot) > 0:
            plot_capture_ratio_by_phase(df_for_plot, os.path.join(out_dir, "capture_ratio_by_phase.png"))
            plot_capture_by_distance_bin(df_for_plot, os.path.join(out_dir, "capture_by_distance_bin.png"))
            plot_distance_heatmap(df_for_plot, os.path.join(out_dir, "capture_heatmap_phase_distance.png"))
        if len(daily_used) >= 2:
            plot_daily_trend_compact(daily_used, os.path.join(out_dir, "daily_capture_trend.png"))
            plot_daily_bin_trend_compact(daily_used, os.path.join(out_dir, "daily_bin_capture_trend.png"))
        print("  Done.")

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Output dir: {out_dir}")
    print(f"  Valid days: {n_used} / {len(daily)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
