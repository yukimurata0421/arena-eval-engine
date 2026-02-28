import os
import sys
from pathlib import Path
import json
import glob
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
    for fp in DIST_FILES:
        if not os.path.exists(fp): continue
        print(f"  Reading: {os.path.basename(fp)}")
        with open_file(fp) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get('src') != 'dist_1m': continue
                    ts = d['ts']
                    dt_jst = datetime.fromtimestamp(ts + 32400)
                    dist_rows.append({
                        'date': dt_jst.date(),
                        'time_bin': get_fast_time_bin(ts),
                        'auc_n_used': d.get('n_used', 0),
                        'n_total': d.get('n_total', 0)
                    })
                except: continue
    
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
    for i, pf in enumerate(sorted(POS_FILES)):
        base = os.path.basename(pf)
        date_str = "".join(filter(str.isdigit, base))[:8]
        try:
            target_date = datetime.strptime(date_str, "%Y%m%d").date()
        except: continue
        
        print(f"  [{i+1}/{len(POS_FILES)}] Processing: {base} ...", end="\r")
        
        hourly_hex = {f"{h:02d}-{(h+BIN_HOURS):02d}": set() for h in range(0, 24, BIN_HOURS)}
        
        try:
            with open_file(pf) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        t_bin = get_fast_time_bin(d['ts'])
                        if t_bin in hourly_hex:
                            hourly_hex[t_bin].add(d['hex'])
                    except: continue
        except Exception as e:
            print(f"\n⚠️ Error ({base}): {e}")
            continue
        
        for t_bin, hex_set in hourly_hex.items():
            traffic_rows.append({'date': target_date, 'time_bin': t_bin, 'traffic_proxy': len(hex_set)})
    print("\n>>> Aircraft density computation complete.")

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
