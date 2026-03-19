"""Shared constants and utility functions for the OpenSky comparison package."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

from arena.lib.config import get_distance_bins_km, get_quality_thresholds, get_site_latlon
from arena.lib.platform_setup import resolve_workers

SITE_LAT, SITE_LON = get_site_latlon()

DISTANCE_BINS_KM = get_distance_bins_km()
DISTANCE_BIN_LABELS = [
    f"{int(DISTANCE_BINS_KM[i])}-{int(DISTANCE_BINS_KM[i + 1])}"
    for i in range(len(DISTANCE_BINS_KM) - 1)
]

DEFAULT_OS_MIN_N_USED = 3
DEFAULT_OS_MAX_KM_MAX = 600.0
_MIN_AUC_N_USED, _MIN_MINUTES_COVERED = get_quality_thresholds()
DEFAULT_OS_MIN_MINUTES_PER_DAY = _MIN_MINUTES_COVERED
DEFAULT_LOCAL_UTC_OFFSET_HOURS = 9
DEFAULT_LOCAL_MIN_MINUTES_MATCH = _MIN_MINUTES_COVERED
DEFAULT_CR_CAP = 5.0

MAX_WORKERS = resolve_workers()


def normalize_path(p: str) -> str:
    if os.name == "nt":
        return p
    m = re.match(r"^([A-Za-z]):[/\\](.*)$", p)
    if not m:
        return p
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def ts_to_utc_minute(ts: float) -> int:
    return int(ts // 60)


def ts_to_date_str(ts: float, utc_offset_hours: int = 0) -> str:
    ts_shifted = float(ts) + (float(utc_offset_hours) * 3600.0)
    return datetime.fromtimestamp(ts_shifted, tz=timezone.utc).strftime("%Y%m%d")


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
