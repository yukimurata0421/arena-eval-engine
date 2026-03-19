"""OpenSky and local ADS-B data loading."""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from arena.lib.geo import haversine_km
from arena.log import get_script_logger

log = get_script_logger(__name__)

from _common import (
    DISTANCE_BIN_LABELS,
    DISTANCE_BINS_KM,
    DEFAULT_LOCAL_UTC_OFFSET_HOURS,
    DEFAULT_OS_MAX_KM_MAX,
    DEFAULT_OS_MIN_N_USED,
    MAX_WORKERS,
    SITE_LAT,
    SITE_LON,
    classify_distance_bin,
    iter_jsonl,
    ts_to_date_str,
    ts_to_utc_minute,
)


# ── OpenSky ──────────────────────────────────────────────────────────

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
    date_utc_offset_hours: int = DEFAULT_LOCAL_UTC_OFFSET_HOURS,
) -> Tuple[List[OpenSkyMinute], Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(opensky_dir, pattern)))
    if not paths:
        single = os.path.join(opensky_dir, "dist_1m.jsonl")
        if os.path.exists(single):
            paths = [single]

    raw_count = 0
    results: List[OpenSkyMinute] = []
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
            results.append(OpenSkyMinute(
                ts=float(ts),
                minute_key=ts_to_utc_minute(float(ts)),
                date_str=ts_to_date_str(float(ts), utc_offset_hours=date_utc_offset_hours),
                n_total=int(rec.get("n_total", 0)),
                n_used=n_used,
                km_n=int(km.get("n", 0)),
                km_avg=float(km.get("avg", 0)),
                km_p50=float(km.get("p50", 0)),
                km_p75=float(km.get("p75", 0)),
                km_p90=float(km.get("p90", 0)),
                km_p95=float(km.get("p95", 0)),
                km_max=km_max_val,
            ))

    quality_report = {
        "raw_records": raw_count,
        "accepted": len(results),
        "rejected_low_n_used": rejected_low_n,
        "rejected_high_km_max": rejected_high_km,
        "min_n_used_threshold": min_n_used,
        "max_km_max_threshold": max_km_max,
    }
    return results, quality_report


# ── Local ADS-B ──────────────────────────────────────────────────────

@dataclass
class LocalMinuteSummary:
    minute_key: int
    date_str: str
    n_unique: int
    km_max: float
    bin_counts: Dict[str, int]


def get_existing_local_dates(local_dir: str, pattern: str = "pos_*.jsonl") -> set:
    dates: set = set()
    for p in glob.glob(os.path.join(local_dir, pattern)):
        m = re.search(r"pos_(\d{8})\.jsonl$", os.path.basename(p))
        if m:
            dates.add(m.group(1))
    return dates


def _process_local_file(
    path: str,
    site_latlon: Tuple[float, float],
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, str], int]:
    minute_hex_dist: Dict[int, Dict[str, float]] = defaultdict(dict)
    minute_dates: Dict[int, str] = {}

    date_from_fn = None
    m = re.search(r"pos_(\d{8})\.jsonl$", os.path.basename(path))
    if m:
        date_from_fn = m.group(1)

    for rec in iter_jsonl(path):
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

    return dict(minute_hex_dist), minute_dates, len(minute_hex_dist)


def load_local_data(
    local_dir: str,
    pattern: str = "pos_*.jsonl",
    target_dates: Optional[set] = None,
    site_latlon: Tuple[float, float] = (SITE_LAT, SITE_LON),
) -> Dict[int, LocalMinuteSummary]:
    paths = sorted(glob.glob(os.path.join(local_dir, pattern)))
    if target_dates:
        filtered = []
        for p in paths:
            m = re.search(r"pos_(\d{8})\.jsonl$", os.path.basename(p))
            if m and m.group(1) in target_dates:
                filtered.append(p)
        paths = filtered

    minute_hex_dist: Dict[int, Dict[str, float]] = defaultdict(dict)
    minute_dates: Dict[int, str] = {}

    if paths:
        log.info(f" [local] {len(paths)} Read files in parallel (workers={MAX_WORKERS}) ...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_process_local_file, p, site_latlon): p for p in paths}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                part_dist, part_dates, minute_count = fut.result()
                log.info(f" [local] completed {os.path.basename(p)} (minutes={minute_count}) ...")
            except Exception as e:
                log.info(f" [local] failure {os.path.basename(p)} ({e})")
                continue
            for mk, hx_map in part_dist.items():
                dst = minute_hex_dist[mk]
                for hx, dist_km in hx_map.items():
                    if hx not in dst or dist_km > dst[hx]:
                        dst[hx] = dist_km
                if mk not in minute_dates and mk in part_dates:
                    minute_dates[mk] = part_dates[mk]

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


# ── Bin estimation ───────────────────────────────────────────────────

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
