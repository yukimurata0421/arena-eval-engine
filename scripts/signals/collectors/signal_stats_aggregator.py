import os
import sys
from pathlib import Path
import json
import pandas as pd
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


from arena.lib.paths import DATA_DIR, PAST_LOG_DIR, ADSB_SIGNAL_RANGE_SUMMARY

SEARCH_DIRS = [str(DATA_DIR), str(PAST_LOG_DIR)]
OUTPUT_FILE = str(ADSB_SIGNAL_RANGE_SUMMARY)

def combine_metrics(buckets, keys, metric_key):
    total_val = 0
    total_samples = 0
    for k in keys:
        b = buckets.get(k, {})
        n = b.get('n_samples', 0)
        v = b.get(metric_key)
        if n > 0 and v is not None:
            total_val += v * n
            total_samples += n
    return total_val / total_samples if total_samples > 0 else None

def _process_file(f_path: str) -> dict:
    if not os.path.isfile(f_path) or f_path.endswith(('.py', '.csv')):
        return {}
    per_date = {}
    # Guard against encoding issues in large JSONL files
    with open(f_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                d = json.loads(line)
                ts = d.get('ts') or (datetime.fromisoformat(d['ts_iso'].replace('Z', '')).timestamp() if 'ts_iso' in d else None)
                if not ts:
                    continue
                dt = datetime.fromtimestamp(ts).date()
                buckets = d.get('buckets', {})

                b_150 = buckets.get('150-175km', {})
                if b_150.get('n_samples', 0) > 0:
                    sig = b_150.get('avg_signal')
                    snr = b_150.get('avg_snr')

                    entry = per_date.setdefault(dt, {"sig_sum": 0.0, "sig_n": 0, "snr_sum": 0.0, "snr_n": 0})
                    if sig is not None:
                        entry["sig_sum"] += float(sig)
                        entry["sig_n"] += 1
                    if snr is not None:
                        entry["snr_sum"] += float(snr)
                        entry["snr_n"] += 1
            except Exception:
                continue
    return per_date


def _merge_into(merged: dict, per_file: dict) -> None:
    for dt, v in per_file.items():
        entry = merged.setdefault(dt, {"sig_sum": 0.0, "sig_n": 0, "snr_sum": 0.0, "snr_n": 0})
        entry["sig_sum"] += v["sig_sum"]
        entry["sig_n"] += v["sig_n"]
        entry["snr_sum"] += v["snr_sum"]
        entry["snr_n"] += v["snr_n"]


def aggregate_signal_ranges():
    files = []
    for d in SEARCH_DIRS:
        if os.path.exists(d):
            files.extend(glob.glob(os.path.join(d, "*signal*")))
    files = list(set(files))

    if not files:
        print(">>> 0 files: signal strength data not found.")
        df = pd.DataFrame(columns=["date", "sig_150_175", "snr_150_175"])
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        return

    print(f">>> Aggregating signal strength and SNR from {len(files)} files... (parallel)")

    max_workers = min(6, os.cpu_count() or 6)

    merged = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_file, f_path) for f_path in files]
        for fut in as_completed(futures):
            try:
                per_file = fut.result()
                _merge_into(merged, per_file)
            except Exception:
                continue

    if not merged and files:
        print(">>> Parallel aggregation was empty. Re-running serially.")
        for f_path in files:
            try:
                per_file = _process_file(f_path)
                _merge_into(merged, per_file)
            except Exception:
                continue

    rows = []
    for dt, v in merged.items():
        row = {"date": dt}
        row["sig_150_175"] = (v["sig_sum"] / v["sig_n"]) if v["sig_n"] > 0 else None
        row["snr_150_175"] = (v["snr_sum"] / v["snr_n"]) if v["snr_n"] > 0 else None
        rows.append(row)

    if not rows:
        df = pd.DataFrame(columns=["date", "sig_150_175", "snr_150_175"])
    else:
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Range aggregation (with SNR) complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    aggregate_signal_ranges()
