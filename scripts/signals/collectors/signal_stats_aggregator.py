import os
import json
import glob
from datetime import date, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from arena.lib.paths import (
    ADSB_SIGNAL_DAILY_SUMMARY,
    ADSB_SIGNAL_RANGE_SUMMARY,
    DATA_DIR,
    PAST_LOG_DIR,
)
from arena.lib.platform_setup import resolve_workers
from arena.log import get_script_logger

try:
    from orjson import loads as _json_loads
except ImportError:
    _json_loads = json.loads  # type: ignore[assignment]

log = get_script_logger(__name__)

SEARCH_DIRS = [str(DATA_DIR), str(PAST_LOG_DIR)]
OUTPUT_FILE = str(ADSB_SIGNAL_RANGE_SUMMARY)
LEGACY_OUTPUT_FILE = str(ADSB_SIGNAL_DAILY_SUMMARY)

BAND_EDGES = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 9999]
N_BANDS = len(BAND_EDGES) - 1
BANDS = [(BAND_EDGES[i], BAND_EDGES[i + 1]) for i in range(N_BANDS)]


def _band_col(lo: int, hi: int, prefix: str = "sig") -> str:
    return f"{prefix}_{lo}_{hi}"


SIG_COLS = [_band_col(b[0], b[1], "sig") for b in BANDS]
SNR_COLS = [_band_col(b[0], b[1], "snr") for b in BANDS]
ALL_COLS = ["date"] + SIG_COLS + SNR_COLS

# 150-175km band index (for legacy CSV)
_SIG150_IDX = next(i for i, b in enumerate(BANDS) if b == (150, 175))


# ------------------------------------------------------------------
# Bucket-key → band-index mapping  (cached per unique bucket set)
# ------------------------------------------------------------------
_mapping_cache: dict[tuple[str, ...], list[list[str]]] = {}


def _parse_src_bucket(key: str) -> tuple[int, int]:
    k = key.replace("km", "").strip()
    if k.endswith("+"):
        return int(k[:-1]), 9999
    lo, hi = k.split("-", 1)
    return int(lo), int(hi)


def _get_band_groups(src_keys: tuple[str, ...]) -> list[list[str]]:
    cached = _mapping_cache.get(src_keys)
    if cached is not None:
        return cached
    groups: list[list[str]] = [[] for _ in range(N_BANDS)]
    for src_key in src_keys:
        try:
            lo, _hi = _parse_src_bucket(src_key)
        except (ValueError, IndexError):
            continue
        for i in range(N_BANDS):
            if BAND_EDGES[i] <= lo < BAND_EDGES[i + 1]:
                groups[i].append(src_key)
                break
    _mapping_cache[src_keys] = groups
    return groups


# ------------------------------------------------------------------
# Per-file processing  (runs in worker process)
# ------------------------------------------------------------------
# Accumulator layout per band: [sig_sum, sig_n, snr_sum, snr_n]
_ACC_LEN = 4


def _new_entry() -> list[list[float]]:
    return [[0.0, 0, 0.0, 0] for _ in range(N_BANDS)]


def _process_file(f_path: str) -> dict[date, list[list[float]]]:
    if not os.path.isfile(f_path) or f_path.endswith((".py", ".csv")):
        return {}

    per_date: dict[date, list[list[float]]] = {}
    band_edges = BAND_EDGES

    with open(f_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = _json_loads(line)
            except Exception:
                continue

            ts = d.get("ts")
            if not ts:
                ts_iso = d.get("ts_iso")
                if ts_iso:
                    ts = datetime.fromisoformat(ts_iso.replace("Z", "")).timestamp()
                else:
                    continue

            buckets = d.get("buckets")
            if not buckets:
                continue

            dt = date.fromtimestamp(ts)
            band_groups = _get_band_groups(tuple(sorted(buckets)))

            entry = per_date.get(dt)
            if entry is None:
                entry = _new_entry()
                per_date[dt] = entry

            for i in range(N_BANDS):
                src_keys = band_groups[i]
                if not src_keys:
                    continue

                # Inline weighted-average across source buckets
                w_sig = 0.0
                w_snr = 0.0
                n_total = 0
                got_sig = False
                got_snr = False

                for sk in src_keys:
                    b = buckets[sk]
                    n = b.get("n_samples", 0)
                    if n <= 0:
                        continue
                    n_total += n
                    sig = b.get("avg_signal")
                    if sig is not None:
                        w_sig += sig * n
                        got_sig = True
                    snr = b.get("avg_snr")
                    if snr is not None:
                        w_snr += snr * n
                        got_snr = True

                if n_total > 0:
                    acc = entry[i]
                    if got_sig:
                        acc[0] += w_sig / n_total
                        acc[1] += 1
                    if got_snr:
                        acc[2] += w_snr / n_total
                        acc[3] += 1

    return per_date


# ------------------------------------------------------------------
# Merge results from multiple files / workers
# ------------------------------------------------------------------
def _merge_into(merged: dict, per_file: dict) -> None:
    for dt, bands in per_file.items():
        entry = merged.get(dt)
        if entry is None:
            merged[dt] = [b[:] for b in bands]
            continue
        for i in range(N_BANDS):
            s, d = bands[i], entry[i]
            d[0] += s[0]
            d[1] += s[1]
            d[2] += s[2]
            d[3] += s[3]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def aggregate_signal_ranges() -> None:
    files: list[str] = []
    for d in SEARCH_DIRS:
        if os.path.exists(d):
            files.extend(glob.glob(os.path.join(d, "*signal*")))
    files = list(set(files))

    if not files:
        log.info(">>> 0 files: No signal strength data found.")
        df = pd.DataFrame(columns=ALL_COLS)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        df[["date", "sig_150_175"]].to_csv(LEGACY_OUTPUT_FILE, index=False)
        return

    log.info(f">>> Aggregating signal strength and SNR from {len(files)} files...")
    merged: dict = {}

    if len(files) <= 2:
        for f_path in files:
            try:
                _merge_into(merged, _process_file(f_path))
            except Exception:
                continue
    else:
        max_workers = resolve_workers(default_cap=12)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_process_file, f) for f in files]
            for fut in as_completed(futures):
                try:
                    _merge_into(merged, fut.result())
                except Exception:
                    continue

    rows: list[dict] = []
    for dt, bands in merged.items():
        row: dict = {"date": dt}
        for i, (lo, hi) in enumerate(BANDS):
            acc = bands[i]
            row[_band_col(lo, hi, "sig")] = (acc[0] / acc[1]) if acc[1] > 0 else None
            row[_band_col(lo, hi, "snr")] = (acc[2] / acc[3]) if acc[3] > 0 else None
        rows.append(row)

    if not rows:
        df = pd.DataFrame(columns=ALL_COLS)
    else:
        df = pd.DataFrame(rows, columns=ALL_COLS).sort_values("date").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    df[["date", "sig_150_175"]].to_csv(LEGACY_OUTPUT_FILE, index=False)

    n_bands = sum(1 for c in SIG_COLS if df[c].notna().any())
    log.info(f"[OK] Range aggregation complete: {n_bands}/{N_BANDS} bands × {len(df)} days → {OUTPUT_FILE}")


if __name__ == "__main__":
    aggregate_signal_ranges()
