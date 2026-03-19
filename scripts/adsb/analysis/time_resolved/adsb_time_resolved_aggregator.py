import os
import sys
from pathlib import Path
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed


from arena.lib.paths import DATA_DIR, RAW_DIR, OUTPUT_DIR

BASE_DIR = str(DATA_DIR)
POS_DIR = os.path.join(BASE_DIR, "plao_pos")
OUTPUT_DIR = os.path.join(str(OUTPUT_DIR), "time_resolved")
from arena.lib.phase_config import get_config as _get_cfg

from arena.log import get_script_logger

log = get_script_logger(__name__)
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


def _process_pos_file(pf):
    """1つの pos ファイルを処理して [{date, time_bin, traffic_proxy}] を返す。
    ProcessPoolExecutor で pickle できるようモジュールレベルに定義。
    """
    base = os.path.basename(pf)
    date_str = "".join(filter(str.isdigit, base))[:8]
    try:
        target_date = datetime.strptime(date_str, "%Y%m%d").date()
    except Exception:
        return []
    hourly_hex = {f"{h:02d}-{(h+BIN_HOURS):02d}": set() for h in range(0, 24, BIN_HOURS)}
    try:
        with open_file(pf) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    t_bin = get_fast_time_bin(d['ts'])
                    if t_bin in hourly_hex:
                        hourly_hex[t_bin].add(d['hex'])
                except Exception:
                    continue
    except Exception as e:
        log.info(f"\n⚠️ エラー ({base}): {e}")
        return []
    return [{'date': target_date, 'time_bin': t_bin, 'traffic_proxy': len(hs)}
            for t_bin, hs in hourly_hex.items()]


def process_aggregator():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log.info(f">>> AUC データ集計中...")
    dist_rows = []
    for fp in DIST_FILES:
        if not os.path.exists(fp): continue
        log.info(f"  読み込み: {os.path.basename(fp)}")
        total_lines = skip_lines = 0
        with open_file(fp) as f:
            for line in f:
                total_lines += 1
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
                except Exception:
                    skip_lines += 1
                    continue
        if total_lines > 0 and skip_lines / total_lines > 0.05:
            log.info(f"  [WARN] {os.path.basename(fp)}: {skip_lines}/{total_lines} 行をスキップしました"
                  f" ({skip_lines/total_lines*100:.1f}%)。データ形式を確認してください。")
    
    if not dist_rows:
        output_path = os.path.join(OUTPUT_DIR, "adsb_timebin_summary.csv")
        pd.DataFrame(columns=["date", "time_bin", "auc_sum", "total_packets", "minutes",
                              "traffic_proxy", "post"]).to_csv(output_path, index=False)
        log.info(f"⚠️ AUC データが空です。空CSVを書き出しました: {output_path}")
        return

    df_dist = pd.DataFrame(dist_rows)
    df_auc = df_dist.groupby(['date', 'time_bin']).agg(
        auc_sum=('auc_n_used', 'sum'),
        total_packets=('n_total', 'sum'),
        minutes=('auc_n_used', 'count')
    ).reset_index()

    log.info(f">>> 航空機密度を計算中（対象: {len(POS_FILES)} ファイル）...")
    traffic_rows = []

    max_workers = int(os.environ.get("ARENA_MAX_WORKERS", os.cpu_count() or 4))
    sorted_files = sorted(POS_FILES)
    if max_workers > 1 and len(sorted_files) > 4:
        # ProcessPoolExecutor で並列処理 (既に opensky 評価が同様のパターンを使用)
        with ProcessPoolExecutor(max_workers=min(max_workers, len(sorted_files))) as ex:
            futures = {ex.submit(_process_pos_file, pf): pf for pf in sorted_files}
            done = 0
            for future in as_completed(futures):
                done += 1
                log.info(f"  [{done}/{len(sorted_files)}] 完了")
                traffic_rows.extend(future.result())
    else:
        for i, pf in enumerate(sorted_files):
            log.info(f"  [{i+1}/{len(sorted_files)}] 処理中: {os.path.basename(pf)} ...")
            traffic_rows.extend(_process_pos_file(pf))
    log.info("\n>>> 航空機密度計算が完了しました。")

    if traffic_rows:
        df_traffic = pd.DataFrame(traffic_rows)
    else:
        df_traffic = pd.DataFrame(columns=["date", "time_bin", "traffic_proxy"])
    
    final_df = pd.merge(df_auc, df_traffic, on=['date', 'time_bin'], how='left')
    _med_traffic = final_df['traffic_proxy'].median()
    if not pd.notna(_med_traffic):
        log.info("  [WARN] traffic_proxy が全て NaN です。fill_val=1.0 を使用します。"
              " pos_*.jsonl ファイルが存在するか確認してください。")
    _fill_traffic = _med_traffic if pd.notna(_med_traffic) else 1.0
    final_df['traffic_proxy'] = final_df['traffic_proxy'].fillna(_fill_traffic)
    final_df['post'] = (pd.to_datetime(final_df['date']) >= pd.Timestamp(INTERVENTION_DATE)).astype(int)
    
    expected_mins = BIN_HOURS * 60
    final_df = final_df[final_df['minutes'] >= expected_mins * 0.9]
    
    output_path = os.path.join(OUTPUT_DIR, "adsb_timebin_summary.csv")
    final_df.to_csv(output_path, index=False)
    log.info(f"集計完了: {output_path}（{len(final_df)} サンプル）")

if __name__ == "__main__":
    process_aggregator()
