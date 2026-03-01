import os
import sys
from pathlib import Path
import json
import glob
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import gzip


from arena.lib.paths import DATA_DIR, RAW_DIR, OUTPUT_DIR

BASE_DIR = str(DATA_DIR)
POS_DIR = os.path.join(BASE_DIR, "plao_pos")
OUTPUT_DIR = os.path.join(str(OUTPUT_DIR), "time_resolved")
from arena.lib.phase_config import get_config as _get_cfg
INTERVENTION_DATE = _get_cfg().time_resolved_date
BIN_HOURS = 2

DIST_FILES = [os.path.join(BASE_DIR, "dist_1m.jsonl")] + \
             glob.glob(os.path.join(str(RAW_DIR), "past_log", "*dist*.jsonl*"))
POS_FILES = glob.glob(os.path.join(POS_DIR, "pos_*.jsonl*"))

logger = logging.getLogger("arena.time_resolved_aggregator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _init_counters():
    return {
        "n_total": 0,
        "n_ok": 0,
        "n_skip": 0,
        "n_err": 0,
        "drop_reasons": {},
        "first_error_sample": None,
    }


def _count_drop(counters, reason, sample=None):
    counters["n_skip"] += 1
    counters["drop_reasons"][reason] = counters["drop_reasons"].get(reason, 0) + 1
    if counters["first_error_sample"] is None and sample is not None:
        counters["first_error_sample"] = sample

def get_fast_time_bin(ts):
    """Compute JST 2-hour bucket label from a UTC timestamp."""
    jst_hour = (int(ts) + 32400) // 3600 % 24
    bin_start = (jst_hour // BIN_HOURS) * BIN_HOURS
    return f"{bin_start:02d}-{(bin_start + BIN_HOURS):02d}"

def open_file(path):
    """Open plain text and gz files transparently."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')

def process_aggregator():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f">>> Aggregating AUC data...")
    dist_rows = []
    dist_counters = _init_counters()
    for fp in DIST_FILES:
        if not os.path.exists(fp):
            continue
        print(f"  Reading: {os.path.basename(fp)}")
        with open_file(fp) as f:
            for line in f:
                try:
                    dist_counters["n_total"] += 1
                    d = json.loads(line)
                    if d.get('src') != 'dist_1m':
                        _count_drop(dist_counters, "src_not_dist_1m")
                        continue
                    ts = d['ts']
                    dt_jst = datetime.fromtimestamp(ts + 32400)
                    dist_rows.append({
                        'date': dt_jst.date(),
                        'time_bin': get_fast_time_bin(ts),
                        'auc_n_used': d.get('n_used', 0),
                        'n_total': d.get('n_total', 0)
                    })
                    dist_counters["n_ok"] += 1
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    dist_counters["n_err"] += 1
                    _count_drop(dist_counters, "parse_error", sample={"error": repr(e), "line": line[:200]})
                    continue
    
    if not dist_rows:
        output_path = os.path.join(OUTPUT_DIR, "adsb_timebin_summary.csv")
        pd.DataFrame(columns=["date", "time_bin", "auc_sum", "total_packets", "minutes",
                              "traffic_proxy", "post"]).to_csv(output_path, index=False)
        print(f"⚠️ AUC data empty; wrote empty CSV: {output_path}")
        return

    df_dist = pd.DataFrame(dist_rows)
    df_auc = df_dist.groupby(['date', 'time_bin']).agg(
        auc_sum=('auc_n_used', 'sum'),
        total_packets=('n_total', 'sum'),
        minutes=('auc_n_used', 'count')
    ).reset_index()

    print(f">>> Computing aircraft density (Target: {len(POS_FILES)} files)...")
    traffic_rows = []
    pos_counters = _init_counters()
    for i, pf in enumerate(sorted(POS_FILES)):
        base = os.path.basename(pf)
        date_str = "".join(filter(str.isdigit, base))[:8]
        try:
            target_date = datetime.strptime(date_str, "%Y%m%d").date()
        except (ValueError, TypeError) as e:
            pos_counters["n_err"] += 1
            _count_drop(pos_counters, "bad_filename_date", sample={"file": base, "error": repr(e)})
            continue
        
        print(f"  [{i+1}/{len(POS_FILES)}] Processing: {base} ...", end="\r")
        
        hourly_hex = {f"{h:02d}-{(h+BIN_HOURS):02d}": set() for h in range(0, 24, BIN_HOURS)}
        
        try:
            with open_file(pf) as f:
                for line in f:
                    pos_counters["n_total"] += 1
                    try:
                        d = json.loads(line)
                        t_bin = get_fast_time_bin(d['ts'])
                        if t_bin in hourly_hex:
                            hourly_hex[t_bin].add(d['hex'])
                        else:
                            _count_drop(pos_counters, "time_bin_out_of_range")
                            continue
                        pos_counters["n_ok"] += 1
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                        pos_counters["n_err"] += 1
                        _count_drop(pos_counters, "record_parse_error", sample={"error": repr(e), "line": line[:200]})
                        continue
        except Exception as e:
            logger.warning("pos file read failed: %s (%s)", base, e)
            pos_counters["n_err"] += 1
            _count_drop(pos_counters, "file_read_error", sample={"file": base, "error": repr(e)})
            continue
        
        for t_bin, hex_set in hourly_hex.items():
            traffic_rows.append({'date': target_date, 'time_bin': t_bin, 'traffic_proxy': len(hex_set)})
    print("\n>>> Aircraft density computation complete.")
    logger.info(
        "dist stats: total=%s ok=%s skip=%s err=%s reasons=%s first_error=%s",
        dist_counters["n_total"],
        dist_counters["n_ok"],
        dist_counters["n_skip"],
        dist_counters["n_err"],
        dist_counters["drop_reasons"],
        dist_counters["first_error_sample"],
    )
    logger.info(
        "pos stats: total=%s ok=%s skip=%s err=%s reasons=%s first_error=%s",
        pos_counters["n_total"],
        pos_counters["n_ok"],
        pos_counters["n_skip"],
        pos_counters["n_err"],
        pos_counters["drop_reasons"],
        pos_counters["first_error_sample"],
    )

    if traffic_rows:
        df_traffic = pd.DataFrame(traffic_rows)
    else:
        df_traffic = pd.DataFrame(columns=["date", "time_bin", "traffic_proxy"])
    
    final_df = pd.merge(df_auc, df_traffic, on=['date', 'time_bin'], how='left')
    final_df['traffic_proxy'] = final_df['traffic_proxy'].fillna(final_df['traffic_proxy'].median() or 1)
    final_df['post'] = (pd.to_datetime(final_df['date']) >= pd.Timestamp(INTERVENTION_DATE)).astype(int)
    
    expected_mins = BIN_HOURS * 60
    final_df = final_df[final_df['minutes'] >= expected_mins * 0.9]
    
    output_path = os.path.join(OUTPUT_DIR, "adsb_timebin_summary.csv")
    final_df.to_csv(output_path, index=False)
    print(f"✅ Aggregation complete: {output_path} ({len(final_df)} samples)")

if __name__ == "__main__":
    process_aggregator()
